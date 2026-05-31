use std::collections::HashMap;

use crate::sc::types::ScValue;
use crate::sc::vector3::Vec3;

type CellKey = (i64, i64, i64);

pub(crate) struct SpatialIndex {
	cell: ScValue,
	min: Vec3,
	min_key: CellKey,
	max_key: CellKey,
	buckets: HashMap<CellKey, Vec<usize>>,
}

impl SpatialIndex {
	pub(crate) fn new<I, F>(indices: I, cell: ScValue, point_for_index: F) -> Self
	where
		I: IntoIterator<Item = usize>,
		F: Fn(usize) -> Vec3,
	{
		let indices: Vec<usize> = indices.into_iter().collect();
		let cell = cell.max(1.0e-9);
		let mut min = Vec3::new(ScValue::INFINITY, ScValue::INFINITY, ScValue::INFINITY);
		for &idx in &indices {
			let p = point_for_index(idx);
			if p.x < min.x { min.x = p.x; }
			if p.y < min.y { min.y = p.y; }
			if p.z < min.z { min.z = p.z; }
		}
		if indices.is_empty() {
			min = Vec3::zero();
		}
		let mut out = Self {
			cell,
			min,
			min_key: (0, 0, 0),
			max_key: (0, 0, 0),
			buckets: HashMap::new(),
		};
		for idx in indices {
			let key = out.key(point_for_index(idx));
			if out.buckets.is_empty() {
				out.min_key = key;
				out.max_key = key;
			} else {
				out.min_key.0 = out.min_key.0.min(key.0);
				out.min_key.1 = out.min_key.1.min(key.1);
				out.min_key.2 = out.min_key.2.min(key.2);
				out.max_key.0 = out.max_key.0.max(key.0);
				out.max_key.1 = out.max_key.1.max(key.1);
				out.max_key.2 = out.max_key.2.max(key.2);
			}
			out.buckets.entry(key).or_default().push(idx);
		}
		out
	}

	pub(crate) fn candidates(&self, center: Vec3, radius: ScValue) -> Vec<usize> {
		if self.buckets.is_empty() {
			return Vec::new();
		}
		let layer = (radius / self.cell).ceil().max(1.0) as i64;
		let (kx, ky, kz) = self.key(center);
		let mut out = Vec::new();
		for dx in -layer..=layer {
			for dy in -layer..=layer {
				for dz in -layer..=layer {
					if let Some(bucket) = self.buckets.get(&(kx + dx, ky + dy, kz + dz)) {
						out.extend(bucket.iter().copied());
					}
				}
			}
		}
		out
	}

	pub(crate) fn nearest_candidates<F>(&self, center: Vec3, point_for_index: F) -> Vec<usize>
	where
		F: Fn(usize) -> Vec3,
	{
		if self.buckets.is_empty() {
			return Vec::new();
		}
		let (kx, ky, kz) = self.key(center);
		let mut out = Vec::new();
		let mut layer = 0_i64;
		let mut best_radius2 = ScValue::INFINITY;
		let max_layer = self.max_layer_from(kx, ky, kz);
		loop {
			let start = out.len();
			for dx in -layer..=layer {
				for dy in -layer..=layer {
					for dz in -layer..=layer {
						if dx.abs().max(dy.abs()).max(dz.abs()) != layer {
							continue;
						}
						if let Some(bucket) = self.buckets.get(&(kx + dx, ky + dy, kz + dz)) {
							out.extend(bucket.iter().copied());
						}
					}
				}
			}
			if out.len() > start {
				for &idx in &out[start..] {
					let d2 = center.distance_squared(point_for_index(idx));
					if d2 < best_radius2 {
						best_radius2 = d2;
					}
				}
			}
			if best_radius2.is_finite() && best_radius2 < (layer as ScValue * self.cell).powi(2) {
				break;
			}
			layer += 1;
			if layer > max_layer {
				break;
			}
		}
		out
	}

	fn max_layer_from(&self, kx: i64, ky: i64, kz: i64) -> i64 {
		(kx - self.min_key.0).abs()
			.max((ky - self.min_key.1).abs())
			.max((kz - self.min_key.2).abs())
			.max((kx - self.max_key.0).abs())
			.max((ky - self.max_key.1).abs())
			.max((kz - self.max_key.2).abs())
	}

	fn key(&self, p: Vec3) -> CellKey {
		(
			((p.x - self.min.x) / self.cell).floor() as i64,
			((p.y - self.min.y) / self.cell).floor() as i64,
			((p.z - self.min.z) / self.cell).floor() as i64,
		)
	}
}
