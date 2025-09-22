use bevy::ecs::component::Component;
use bevy::math::IVec2;
pub use bevy_app_compute::prelude::{
    AppComputeWorkerBuilder, ComputeShader, ComputeWorker, ShaderRef,
};

pub mod chunk_generator;

#[derive(Component, Debug)]
pub struct Chunk<T>(pub IVec2, std::marker::PhantomData<T>);