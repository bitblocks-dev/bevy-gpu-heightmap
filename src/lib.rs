use bevy::ecs::component::Component;
use bevy::math::IVec3;
pub use bevy_app_compute::prelude::{ComputeShader, ShaderRef};

pub mod chunk_generator;

#[derive(Component, Debug)]
pub struct Chunk<T>(pub IVec3, std::marker::PhantomData<T>);
