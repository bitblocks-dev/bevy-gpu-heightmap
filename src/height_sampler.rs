use crate::chunk_generator::SampleContext;
use crate::terrain_sampler::DensitySampler;

pub struct HeightDensitySampler<T>(pub T);

impl<T: HeightSampler> DensitySampler for HeightDensitySampler<T> {
    fn sample_density(&self, context: SampleContext<Self>) -> f32 {
        let threshold = context.generator.surface_threshold;
        let world_y = context.world_position.y;
        let height = self.0.sample_height(context);
        threshold + (height - world_y)
    }
}

pub trait HeightSampler {
    fn sample_height(&self, context: SampleContext<HeightDensitySampler<Self>>) -> f32
    where
        Self: Sized;
}

#[cfg(feature = "noise_sampler")]
pub struct NoiseHeightSampler(pub fastnoise_lite::FastNoiseLite);

impl HeightSampler for NoiseHeightSampler {
    fn sample_height(&self, context: SampleContext<HeightDensitySampler<Self>>) -> f32 {
        self.0
            .get_noise_2d(context.world_position.x, context.world_position.z)
            * 0.5
            + 0.5
    }
}

impl Default for NoiseHeightSampler {
    fn default() -> Self {
        Self(fastnoise_lite::FastNoiseLite::new())
    }
}
