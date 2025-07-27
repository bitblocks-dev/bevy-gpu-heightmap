use crate::chunk_generator::SampleContext;
use crate::terrain_sampler::DensitySampler;

pub struct HeightDensitySampler<T>(pub T);

impl<H: HeightSampler> DensitySampler for HeightDensitySampler<H> {
    fn sample_density<T>(&self, context: SampleContext<T>) -> f32 {
        let threshold = context.generator.surface_threshold;
        let world_y = context.world_position.y;
        let height = self.0.sample_height(context);
        threshold + (height - world_y)
    }
}

pub struct RadiusDensitySampler<T>(pub T);

impl<H: HeightSampler> DensitySampler for RadiusDensitySampler<H> {
    fn sample_density<T>(&self, context: SampleContext<T>) -> f32 {
        let threshold = context.generator.surface_threshold;
        let world_y = context.world_position.length();
        let height = self.0.sample_height(context);
        threshold + (height - world_y)
    }
}

pub trait HeightSampler {
    fn sample_height<T>(&self, context: SampleContext<T>) -> f32
    where
        Self: Sized;

    fn scaled(self, scale: f32) -> ScaleHeightSampler<Self>
    where
        Self: Sized,
    {
        ScaleHeightSampler(self, scale)
    }

    fn offset(self, offset: f32) -> OffsetHeightSampler<Self>
    where
        Self: Sized,
    {
        OffsetHeightSampler(self, offset)
    }

    fn build_density(self) -> HeightDensitySampler<Self>
    where
        Self: Sized,
    {
        HeightDensitySampler(self)
    }

    fn build_radius_density(self) -> RadiusDensitySampler<Self>
    where
        Self: Sized,
    {
        RadiusDensitySampler(self)
    }
}

#[cfg(feature = "noise_sampler")]
pub struct NoiseHeightSampler(pub fastnoise_lite::FastNoiseLite);

impl HeightSampler for NoiseHeightSampler {
    fn sample_height<T>(&self, context: SampleContext<T>) -> f32 {
        self.0.get_noise_3d(
            context.world_position.x,
            context.world_position.y,
            context.world_position.z,
        ) * 0.5
            + 0.5
    }
}

impl Default for NoiseHeightSampler {
    fn default() -> Self {
        Self(fastnoise_lite::FastNoiseLite::new())
    }
}

pub struct ScaleHeightSampler<T>(pub T, pub f32);

impl<H: HeightSampler> HeightSampler for ScaleHeightSampler<H> {
    fn sample_height<T>(&self, context: SampleContext<T>) -> f32 {
        self.0.sample_height(context) * self.1
    }
}

pub struct OffsetHeightSampler<T>(pub T, pub f32);

impl<H: HeightSampler> HeightSampler for OffsetHeightSampler<H> {
    fn sample_height<T>(&self, context: SampleContext<T>) -> f32 {
        self.0.sample_height(context) + self.1
    }
}
