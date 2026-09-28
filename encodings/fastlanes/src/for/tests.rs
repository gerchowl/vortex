// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::LazyLock;

use rstest::rstest;
use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::assert_arrays_eq;
use vortex_array::compute::conformance::consistency::test_array_consistency;
use vortex_array::dtype::NativePType;
use vortex_array::session::ArraySessionExt;
use vortex_buffer::Buffer;
use vortex_error::VortexResult;
use vortex_session::VortexSession;

use crate::FL_CHUNK_SIZE;
use crate::FoR;
use crate::FoRArray;
use crate::FoRArrayExt;
use crate::FoRArraySlotsExt;

static SESSION: LazyLock<VortexSession> = LazyLock::new(|| {
    let session = vortex_array::array_session();
    crate::initialize(&session);
    session
});

/// Builds a FoR array over `encoded` with the given per-chunk references, and the values it
/// should decode to.
fn chunked<T: NativePType + num_traits::WrappingAdd>(
    encoded: PrimitiveArray,
    references: &[T],
) -> VortexResult<(FoRArray, PrimitiveArray)> {
    let mut ctx = SESSION.create_execution_ctx();
    let expected = PrimitiveArray::from_option_iter(
        (0..encoded.len())
            .map(|i| {
                let value = encoded.as_slice::<T>()[i];
                let valid = encoded.is_valid(i, &mut ctx)?;
                Ok(valid.then(|| value.wrapping_add(&references[i / FL_CHUNK_SIZE])))
            })
            .collect::<VortexResult<Vec<_>>>()?,
    );
    let expected = if encoded.dtype().is_nullable() {
        expected
    } else {
        PrimitiveArray::new(expected.to_buffer::<T>(), encoded.validity()?)
    };
    let references = Buffer::copy_from(references).into_array();
    Ok((
        FoR::try_new_chunked(encoded.into_array(), references, 0)?,
        expected,
    ))
}

fn unsigned() -> VortexResult<(FoRArray, PrimitiveArray)> {
    chunked(
        PrimitiveArray::from_iter((0..3000u32).map(|i| i % 100)),
        &[0u32, 1_000_000, 5],
    )
}

fn signed_wrapping() -> VortexResult<(FoRArray, PrimitiveArray)> {
    chunked(
        PrimitiveArray::from_iter((0..2100i16).map(|i| i % 7)),
        &[i16::MIN, -3, i16::MAX],
    )
}

fn nullable() -> VortexResult<(FoRArray, PrimitiveArray)> {
    chunked(
        PrimitiveArray::from_option_iter((0..2500i64).map(|i| (i % 5 != 0).then_some(i % 11))),
        &[-1_000i64, 0, 1 << 40],
    )
}

#[rstest]
#[case::unsigned(unsigned())]
#[case::signed_wrapping(signed_wrapping())]
#[case::nullable(nullable())]
fn decodes_per_chunk(#[case] arrays: VortexResult<(FoRArray, PrimitiveArray)>) -> VortexResult<()> {
    let (array, expected) = arrays?;
    assert!(array.constant_reference().is_none());
    assert_arrays_eq!(array, expected, &mut SESSION.create_execution_ctx());
    Ok(())
}

#[rstest]
#[case::unsigned(unsigned())]
#[case::signed_wrapping(signed_wrapping())]
#[case::nullable(nullable())]
fn consistency(#[case] arrays: VortexResult<(FoRArray, PrimitiveArray)>) -> VortexResult<()> {
    let (array, _) = arrays?;
    test_array_consistency(&array.into_array(), &mut SESSION.create_execution_ctx());
    Ok(())
}

#[rstest]
#[case::within_first_chunk(3, 900)]
#[case::across_chunks(1000, 2100)]
#[case::chunk_aligned(1024, 2048)]
#[case::last_chunk(2048, 3000)]
#[case::empty_unaligned(1500, 1500)]
fn slice_keeps_chunk_alignment(#[case] start: usize, #[case] end: usize) -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let (array, expected) = unsigned()?;
    let sliced = array.into_array().slice(start..end)?;
    assert_arrays_eq!(sliced, expected.into_array().slice(start..end)?, &mut ctx);

    if let Some(sliced) = sliced.as_opt::<FoR>() {
        assert_eq!(usize::from(sliced.offset()), start % FL_CHUNK_SIZE);
        // Slicing a slice composes offsets.
        let len = end - start;
        let inner: ArrayRef = sliced.array().slice(len / 3..len)?;
        let expected = unsigned()?.1.into_array().slice(start + len / 3..end)?;
        assert_arrays_eq!(inner, expected, &mut ctx);
    }
    Ok(())
}

#[test]
fn constant_references_keep_the_scalar_reference() -> VortexResult<()> {
    let array = FoR::try_new(
        PrimitiveArray::from_iter(0..3000u32).into_array(),
        7u32.into(),
    )?;
    assert_eq!(array.references().len(), 3);
    assert_eq!(array.constant_reference(), Some(7u32.into()));
    let sliced = array.into_array().slice(1500..1600)?;
    assert_eq!(sliced.as_::<FoR>().constant_reference(), Some(7u32.into()));
    Ok(())
}

#[test]
fn varying_references_do_not_serialize_as_v1() -> VortexResult<()> {
    let (array, _) = unsigned()?;
    assert!(SESSION.array_serialize(array.as_array()).is_err());
    Ok(())
}
