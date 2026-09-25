// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt::Debug;
use std::fmt::Display;
use std::fmt::Formatter;
use std::hash::Hash;
use std::hash::Hasher;
use std::ops::Range;
use std::sync::Arc;

use vortex_buffer::BitBuffer;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;
use vortex_error::vortex_panic;
use vortex_mask::Mask;
use vortex_session::VortexSession;
use vortex_session::registry::CachedId;

use super::Probe;
use crate::ArrayEq;
use crate::ArrayHash;
use crate::ArrayParts;
use crate::ArrayRef;
use crate::EqMode;
use crate::ExecutionCtx;
use crate::ExecutionResult;
use crate::IntoArray;
use crate::array::Array;
use crate::array::ArrayId;
use crate::array::ArrayView;
use crate::array::OperationsVTable;
use crate::array::VTable;
use crate::array::ValidityVTable;
use crate::array::with_empty_buffers;
use crate::arrays::BoolArray;
use crate::arrays::ConstantArray;
use crate::arrays::ScalarFn;
use crate::arrays::constant::list_scalar_elements;
use crate::arrays::filter::FilterReduce;
use crate::arrays::filter::FilterReduceAdaptor;
use crate::arrays::scalar_fn::ExactScalarFn;
use crate::arrays::scalar_fn::ScalarFnArrayExt;
use crate::arrays::scalar_fn::ScalarFnArrayView;
use crate::arrays::slice::SliceReduce;
use crate::arrays::slice::SliceReduceAdaptor;
use crate::buffer::BufferHandle;
use crate::dtype::DType;
use crate::optimizer::rules::ArrayParentReduceRule;
use crate::optimizer::rules::ParentRuleSet;
use crate::scalar::Scalar;
use crate::scalar_fn::fns::list_contains::ListContains;
use crate::scalar_fn::fns::list_contains::ListContainsOptions;
use crate::scalar_fn::fns::list_contains::compute_contains_scalar;
use crate::serde::ArrayChildren;
use crate::validity::Validity;

/// A [`PreparedSet`]-encoded array.
pub type PreparedSetArray = Array<PreparedSet>;

/// A constant, non-null list whose elements are prepared as a set for membership probes.
///
/// Every row holds the same list, as in a [`ConstantArray`]. When [`ListContains`] gets a constant
/// list and a needle that is not canonical, it puts this array in place of the list and gives the
/// node back to the executor. Thus a [`ListContainsElementKernel`] of the needle encoding gets the
/// prepared set as its list, and can probe its own values with [`PreparedSetData::contains`].
///
/// The probe is shared, so a slice or a filter of this array does not build it again, and a
/// constant needle folds against it at optimization. This encoding exists only during execution,
/// and it cannot be serialized.
///
/// [`ListContains`]: crate::scalar_fn::fns::list_contains::ListContains
/// [`ListContainsElementKernel`]: crate::scalar_fn::fns::list_contains::ListContainsElementKernel
#[derive(Clone, Debug)]
pub struct PreparedSet;

/// The data of a [`PreparedSetArray`]: the list, and the probe built from its elements.
#[derive(Clone)]
pub struct PreparedSetData {
    list: Scalar,
    pub(super) set: Arc<ElementSet>,
}

/// The non-null elements of the list in a probe structure, and the facts about the list that
/// decide the answer when no element matches.
pub(super) struct ElementSet {
    pub(super) probe: Probe,
    /// Whether the list holds a null element, which a probe does not hold.
    has_null_element: bool,
    /// Whether the list holds no element at all, counting null elements.
    is_empty: bool,
}

impl PreparedSetData {
    /// Prepares the elements of the non-null list scalar `list`.
    pub(super) fn try_new(list: Scalar, ctx: &mut ExecutionCtx) -> VortexResult<Self> {
        vortex_ensure!(
            matches!(list.dtype(), DType::List(..)),
            "A prepared set needs a list, got {}",
            list.dtype()
        );
        vortex_ensure!(!list.is_null(), "A prepared set needs a non-null list");

        let elements = list_scalar_elements(&list.as_list(), ctx.allocator());
        let is_empty = elements.is_empty();

        // A null element never equals a needle, so the probe is better off without it.
        let valid = elements.validity()?.execute_mask(elements.len(), ctx)?;
        let has_null_element = !valid.all_true();
        let elements = if has_null_element {
            elements.filter(valid)?
        } else {
            elements
        };

        let probe = Probe::try_new(elements, ctx)?;

        Ok(Self {
            list,
            set: Arc::new(ElementSet {
                probe,
                has_null_element,
                is_empty,
            }),
        })
    }

    /// The list that every row holds.
    pub fn list(&self) -> &Scalar {
        &self.list
    }

    /// Whether each of `needles` is an element of the list, under `options`.
    ///
    /// The needles must have the dtype of the list's elements, ignoring nullability. The result
    /// has one row per needle, and the nullability that [`ListContainsOptions::result_nullability`]
    /// declares. A null needle gives `null`. The exception is an empty list off SQL null
    /// semantics, which gives `false` for every needle. Under SQL null semantics, a list that holds
    /// a null element gives `null` for a needle that matches no element.
    pub fn contains(
        &self,
        needles: &ArrayRef,
        options: &ListContainsOptions,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrayRef> {
        let DType::List(element_dtype, _) = self.list.dtype() else {
            vortex_panic!("A prepared set always holds a list");
        };
        if !element_dtype.eq_ignore_nullability(needles.dtype()) {
            vortex_bail!(
                "Element type {} of list does not match search value {}",
                element_dtype,
                needles.dtype(),
            );
        }

        let (bits, needle_validity) = self.set.probe.contains(needles, ctx)?;
        self.finish(bits, needle_validity, needles.dtype(), options)
    }

    /// Makes the result from one membership bit per needle and the validity of the needles.
    fn finish(
        &self,
        bits: BitBuffer,
        needle_validity: Validity,
        needle_dtype: &DType,
        options: &ListContainsOptions,
    ) -> VortexResult<ArrayRef> {
        let nullability = options.result_nullability(self.list.dtype(), needle_dtype);

        let validity = if self.set.is_empty && !options.sql_null_semantics {
            Validity::NonNullable
        } else if options.sql_null_semantics && self.set.has_null_element {
            // Only a match is known. A comparison with the null element makes a non-match unknown.
            needle_validity.and(Validity::from(bits.clone()))?
        } else {
            needle_validity
        };

        Ok(BoolArray::new(bits, validity.union_nullability(nullability)).into_array())
    }
}

impl Debug for PreparedSetData {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PreparedSetData")
            .field("list", &self.list)
            .finish_non_exhaustive()
    }
}

impl Display for PreparedSetData {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "list: {}", self.list)
    }
}

impl ArrayHash for PreparedSetData {
    fn array_hash<H: Hasher>(&self, state: &mut H, _accuracy: EqMode) {
        self.list.hash(state);
    }
}

impl ArrayEq for PreparedSetData {
    fn array_eq(&self, other: &Self, _accuracy: EqMode) -> bool {
        self.list == other.list
    }
}

impl Array<PreparedSet> {
    /// Prepares the non-null list scalar `list` as a set, repeated `len` times.
    pub(crate) fn try_new(list: Scalar, len: usize, ctx: &mut ExecutionCtx) -> VortexResult<Self> {
        let data = PreparedSetData::try_new(list, ctx)?;
        Ok(Self::from_data(data, len))
    }

    /// An array of `len` rows that share the prepared set `data`.
    fn from_data(data: PreparedSetData, len: usize) -> Self {
        let dtype = data.list.dtype().clone();

        // SAFETY: the dtype is the dtype of the list that every row holds.
        unsafe { Array::from_parts_unchecked(ArrayParts::new(PreparedSet, dtype, len, data)) }
    }
}

const PARENT_RULES: ParentRuleSet<PreparedSet> = ParentRuleSet::new(&[
    ParentRuleSet::lift(&ConstantNeedleRule),
    ParentRuleSet::lift(&FilterReduceAdaptor(PreparedSet)),
    ParentRuleSet::lift(&SliceReduceAdaptor(PreparedSet)),
]);

/// Folds `list_contains` of a constant needle against the prepared set into its constant answer.
///
/// [`ListContains::reduce`] folds a [`ConstantArray`] list the same way. This rule covers the list
/// after it is prepared, which the generic constant check does not see.
#[derive(Debug)]
struct ConstantNeedleRule;

impl ArrayParentReduceRule<PreparedSet> for ConstantNeedleRule {
    type Parent = ExactScalarFn<ListContains>;

    fn reduce_parent(
        &self,
        array: ArrayView<'_, PreparedSet>,
        parent: ScalarFnArrayView<'_, ListContains>,
        child_idx: usize,
    ) -> VortexResult<Option<ArrayRef>> {
        // The prepared set is the list child. As the needle it is a list of lists, which this rule
        // does not fold.
        if child_idx != 0 {
            return Ok(None);
        }
        let scalar_fn_array = parent
            .as_opt::<ScalarFn>()
            .vortex_expect("ExactScalarFn matcher confirmed ScalarFnArray");
        let Some(needle) = scalar_fn_array.get_child(1).as_constant() else {
            return Ok(None);
        };

        let result = compute_contains_scalar(array.list(), &needle, parent.options)?;
        Ok(Some(
            ConstantArray::new(result, scalar_fn_array.len()).into_array(),
        ))
    }
}

impl VTable for PreparedSet {
    type TypedArrayData = PreparedSetData;

    type OperationsVTable = Self;
    type ValidityVTable = Self;

    fn id(&self) -> ArrayId {
        static ID: CachedId = CachedId::new("vortex.list.prepared_set");
        *ID
    }

    fn validate(
        &self,
        data: &PreparedSetData,
        dtype: &DType,
        _len: usize,
        _slots: &[Option<ArrayRef>],
    ) -> VortexResult<()> {
        vortex_ensure!(
            data.list.dtype() == dtype,
            "PreparedSetArray list dtype does not match outer dtype"
        );
        Ok(())
    }

    fn nbuffers(_array: ArrayView<'_, Self>) -> usize {
        0
    }

    fn buffer(_array: ArrayView<'_, Self>, idx: usize) -> BufferHandle {
        vortex_panic!("PreparedSetArray buffer index {idx} out of bounds")
    }

    fn buffer_name(_array: ArrayView<'_, Self>, _idx: usize) -> Option<String> {
        None
    }

    fn with_buffers(
        &self,
        array: ArrayView<'_, Self>,
        buffers: &[BufferHandle],
    ) -> VortexResult<ArrayParts<Self>> {
        with_empty_buffers(self, array, buffers)
    }

    fn slot_name(_array: ArrayView<'_, Self>, idx: usize) -> String {
        vortex_panic!("PreparedSetArray slot_name index {idx} out of bounds")
    }

    fn serialize(
        _array: ArrayView<'_, Self>,
        _session: &VortexSession,
    ) -> VortexResult<Option<Vec<u8>>> {
        vortex_bail!("PreparedSetArray is not serializable")
    }

    fn deserialize(
        &self,
        _dtype: &DType,
        _len: usize,
        _metadata: &[u8],
        _buffers: &[BufferHandle],
        _children: &dyn ArrayChildren,
        _session: &VortexSession,
    ) -> VortexResult<ArrayParts<Self>> {
        vortex_bail!("PreparedSetArray is not serializable")
    }

    fn execute(array: Array<Self>, _ctx: &mut ExecutionCtx) -> VortexResult<ExecutionResult> {
        // The rows hold the list only. The probe is of no use to the canonical form.
        Ok(ExecutionResult::done(ConstantArray::new(
            array.data().list.clone(),
            array.len(),
        )))
    }

    fn reduce_parent(
        array: ArrayView<'_, Self>,
        parent: &ArrayRef,
        child_idx: usize,
    ) -> VortexResult<Option<ArrayRef>> {
        PARENT_RULES.evaluate(array, parent, child_idx)
    }
}

impl OperationsVTable<PreparedSet> for PreparedSet {
    type ProbeState = ();

    fn scalar_at(
        array: ArrayView<'_, PreparedSet>,
        _index: usize,
        _ctx: &mut ExecutionCtx,
    ) -> VortexResult<Scalar> {
        Ok(array.list.clone())
    }
}

impl ValidityVTable<PreparedSet> for PreparedSet {
    fn validity(_array: ArrayView<'_, PreparedSet>) -> VortexResult<Validity> {
        // The list is never null.
        Ok(Validity::AllValid)
    }
}

impl SliceReduce for PreparedSet {
    fn slice(array: ArrayView<'_, Self>, range: Range<usize>) -> VortexResult<Option<ArrayRef>> {
        Ok(Some(
            PreparedSetArray::from_data(array.data().clone(), range.len()).into_array(),
        ))
    }
}

impl FilterReduce for PreparedSet {
    fn filter(array: ArrayView<'_, Self>, mask: &Mask) -> VortexResult<Option<ArrayRef>> {
        Ok(Some(
            PreparedSetArray::from_data(array.data().clone(), mask.true_count()).into_array(),
        ))
    }
}
