use std::any::type_name;
use std::fmt::{Debug, Formatter};

use bevy::prelude::*;

use crate::chunk_generator::SampleContext;

pub trait DensitySampler {
    fn sample_density<T>(&self, context: SampleContext<T>) -> f32;
}

#[derive(Debug)]
pub struct BoxDensitySampler {
    pub center: Vec3,
    pub size: Vec3,
    pub rotation: Quat,
}

impl DensitySampler for BoxDensitySampler {
    fn sample_density<T>(&self, context: SampleContext<T>) -> f32 {
        let position = context.world_position - self.center;
        let position = self.rotation * position;
        let distance = position.abs();
        let normalized_distance = distance / (self.size * 0.5);
        let max_distance = normalized_distance.max_element();
        1.0 - max_distance
    }
}

impl Default for BoxDensitySampler {
    fn default() -> Self {
        Self {
            center: Vec3::splat(16.0),
            size: Vec3::splat(8.0),
            rotation: Quat::IDENTITY,
        }
    }
}

#[cfg(feature = "noise_sampler")]
pub struct NoiseDensitySampler(pub fastnoise_lite::FastNoiseLite);

impl Debug for NoiseDensitySampler {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct(type_name::<Self>()).finish()
    }
}

impl DensitySampler for NoiseDensitySampler {
    fn sample_density<T>(&self, context: SampleContext<T>) -> f32 {
        self.0.get_noise_3d(
            context.world_position.x,
            context.world_position.y,
            context.world_position.z,
        ) * 0.5
            + 0.5
    }
}

impl Default for NoiseDensitySampler {
    fn default() -> Self {
        Self(fastnoise_lite::FastNoiseLite::new())
    }
}
