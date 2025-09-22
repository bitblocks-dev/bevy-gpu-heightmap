use bevy::color::palettes::css;
use bevy::pbr::wireframe::{WireframeConfig, WireframePlugin};
use bevy::prelude::*;
use bevy_gpu_heightmap::chunk_generator::{
    ChunkGeneratorSettings, ChunkLoader, ChunkMaterial, HeightmapPlugin,
};
use bevy::asset::embedded_asset;

fn main() {
    App::new()
        .add_plugins((
            DefaultPlugins,
            WireframePlugin::default(),
            bevy_panorbit_camera::PanOrbitCameraPlugin,
            NoiseExamplePlugin,
        ))
        .insert_resource(WireframeConfig {
            global: true,
            default_color: css::WHITE.into(),
        })
        .add_systems(Startup, setup)
        .run();
}

pub struct NoiseExamplePlugin;
impl Plugin for NoiseExamplePlugin {
    fn build(&self, app: &mut App) {
        debug!("Building noise example plugin");
        let (heightmap_handle, material_handle) = {
            let world = app.world_mut();
            let heightmap_handle = ;
            let mut materials: Mut<'_, Assets<StandardMaterial>> = world.resource_mut::<Assets<StandardMaterial>>();
            let material_handle = materials.add(Color::from(tailwind::EMERALD_500));
            (heightmap_handle, material_handle)
        };

        app
            .insert_resource(ChunkGeneratorSettings::new(heightmap_handle, 32, 8))
            .insert_resource(ChunkMaterial::<StandardMaterial>::new(material_handle))
            .add_plugins(HeightmapPlugin::<StandardMaterial>::default());
    }
}

fn setup(mut commands: Commands) {
    debug!("Starting noise example");
    commands.spawn((
        Name::new("Camera"),
        Camera3d::default(),
        Transform::from_xyz(4.0, 6.5, 8.0).looking_at(Vec3::ZERO, Vec3::Y),
        bevy_panorbit_camera::PanOrbitCamera {
            button_orbit: MouseButton::Left,
            button_pan: MouseButton::Left,
            modifier_pan: Some(KeyCode::ShiftLeft),
            reversed_zoom: true,
            ..default()
        },
    ));

    commands.spawn((
        Name::new("Light"),
        DirectionalLight {
            illuminance: 4000.0,
            ..default()
        },
        Transform {
            rotation: Quat::from_euler(EulerRot::XYZ, -1.9, 0.8, 0.0),
            ..default()
        },
    ));

    commands.spawn(ChunkLoader::new(1));
}
