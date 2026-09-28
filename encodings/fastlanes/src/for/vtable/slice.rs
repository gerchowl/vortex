// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ops::Range;

use vortex_array::ArrayRef;
use vortex_array::ArrayView;
use vortex_array::IntoArray;
use vortex_array::arrays::slice::SliceReduce;
use vortex_error::VortexResult;

use crate::FL_CHUNK_SIZE;
use crate::FoR;
use crate::r#for::array::FoRArrayExt;
use crate::r#for::array::FoRArraySlotsExt;

impl SliceReduce for FoR {
    fn slice(array: ArrayView<'_, Self>, range: Range<usize>) -> VortexResult<Option<ArrayRef>> {
        if let Some(reference) = array.constant_reference() {
            return Ok(Some(
                FoR::try_new(array.encoded().slice(range)?, reference)?.into_array(),
            ));
        }

        let start = usize::from(array.offset()) + range.start;
        let end = usize::from(array.offset()) + range.end;
        let references = array
            .references()
            .slice(start / FL_CHUNK_SIZE..end.div_ceil(FL_CHUNK_SIZE))?;
        let offset = u16::try_from(start % FL_CHUNK_SIZE)?;
        Ok(Some(
            FoR::try_new_chunked(array.encoded().slice(range)?, references, offset)?.into_array(),
        ))
    }
}
