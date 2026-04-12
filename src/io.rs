//! Simple binary I/O for region embeddings.
//!
//! Format (little-endian):
//! - Header: `n: u32`, `dim: u32`
//! - For each of n regions: `id: u32`, `min: [f32; dim]`, `max: [f32; dim]`
//!
//! This is intentionally minimal -- no compression, no versioning.
//! For production use cases, prefer serde with a format of your choice.

use crate::{AxisBox, Region};
use std::io::{self, Read, Write};

/// Write a collection of boxes to a binary stream.
pub fn write_boxes<W: Write>(writer: &mut W, ids: &[u32], boxes: &[AxisBox]) -> io::Result<()> {
    assert_eq!(ids.len(), boxes.len());
    if boxes.is_empty() {
        writer.write_all(&0u32.to_le_bytes())?;
        writer.write_all(&0u32.to_le_bytes())?;
        return Ok(());
    }

    let n = boxes.len() as u32;
    let dim = boxes[0].dim() as u32;
    writer.write_all(&n.to_le_bytes())?;
    writer.write_all(&dim.to_le_bytes())?;

    for (id, b) in ids.iter().zip(boxes.iter()) {
        writer.write_all(&id.to_le_bytes())?;
        for &v in b.min() {
            writer.write_all(&v.to_le_bytes())?;
        }
        for &v in b.max() {
            writer.write_all(&v.to_le_bytes())?;
        }
    }

    Ok(())
}

/// Read boxes from a binary stream.
///
/// Returns `(ids, boxes)`.
pub fn read_boxes<R: Read>(reader: &mut R) -> io::Result<(Vec<u32>, Vec<AxisBox>)> {
    let mut buf4 = [0u8; 4];

    reader.read_exact(&mut buf4)?;
    let n = u32::from_le_bytes(buf4) as usize;
    reader.read_exact(&mut buf4)?;
    let dim = u32::from_le_bytes(buf4) as usize;

    let mut ids = Vec::with_capacity(n);
    let mut boxes = Vec::with_capacity(n);

    let mut float_buf = vec![0u8; dim * 4];

    for _ in 0..n {
        reader.read_exact(&mut buf4)?;
        ids.push(u32::from_le_bytes(buf4));

        reader.read_exact(&mut float_buf)?;
        let min: Vec<f32> = float_buf
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();

        reader.read_exact(&mut float_buf)?;
        let max: Vec<f32> = float_buf
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();

        boxes.push(AxisBox::new(min, max));
    }

    Ok((ids, boxes))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip() {
        let ids = vec![0, 1, 2];
        let boxes = vec![
            AxisBox::new(vec![0.0, 0.0], vec![1.0, 1.0]),
            AxisBox::new(vec![2.0, 3.0], vec![4.0, 5.0]),
            AxisBox::new(vec![-1.0, -1.0], vec![0.0, 0.0]),
        ];

        let mut buf = Vec::new();
        write_boxes(&mut buf, &ids, &boxes).unwrap();

        let (read_ids, read_boxes) = read_boxes(&mut buf.as_slice()).unwrap();
        assert_eq!(ids, read_ids);
        assert_eq!(read_boxes.len(), 3);

        for (orig, read) in boxes.iter().zip(read_boxes.iter()) {
            assert_eq!(orig.min(), read.min());
            assert_eq!(orig.max(), read.max());
        }
    }

    #[test]
    fn empty() {
        let mut buf = Vec::new();
        write_boxes(&mut buf, &[], &[]).unwrap();
        let (ids, boxes) = read_boxes(&mut buf.as_slice()).unwrap();
        assert!(ids.is_empty());
        assert!(boxes.is_empty());
    }
}
