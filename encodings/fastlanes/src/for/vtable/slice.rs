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
        // Every chunk shares one reference, so the result needs no offset.
        if let Some(reference) = array.constant_reference() {
            return Ok(Some(
                FoR::try_new(array.encoded().slice(range)?, reference)?.into_array(),
            ));
        }

        // Slicing is zero-copy: `encoded` is sliced by row, `references` by chunk, and the new
        // `offset` records where the first remaining row sits within its chunk.
        //
        // `start` and `end` are the requested range measured from the start of the first chunk,
        // so the array's own `offset` is added to it. The chunks overlapping `start..end` are
        // kept, and the new offset is the position of `start` within its chunk.
        //
        // For example, an array with `offset = 476` and references `[r1, r2]` (itself the slice
        // `1500..2500` of a 3000-row array with references `[r0, r1, r2]`) sliced to `600..900`:
        //
        //   start      = 476 + 600 = 1076
        //   end        = 476 + 900 = 1376
        //   references = [r1, r2].slice(1076 / 1024 .. ceil(1376 / 1024))
        //              = [r1, r2].slice(1..2) = [r2]
        //   offset     = 1076 % 1024 = 52
        //
        // Row 0 of the result is row 2100 of the original array, which is 52 rows into the
        // original chunk 2 (rows 2048..3000), whose reference is `r2`.
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
