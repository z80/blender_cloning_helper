# Blender Cloning Helper

A Blender addon that integrates COLMAP photogrammetry with powerful tools for texture painting, image alignment, and UDIM material creation. Designed to simplify photogrammetry workflows and enable accurate 3D reconstruction from video-derived frames.

## Features

- COLMAP data import: camera poses, sparse point cloud, and image planes
- Camera alignment: snap Blender's view to a selected reference image
- Stencil texture generation: create a paint-ready texture aligned to reference images
- Automatic UDIM material builder
- Built-in FFmpeg support: extract frames from video input (e.g., phone selfie)

## Visual Overview

Reference image placement  
![Scene Reference](docs/images/reference_viewport.png)

Stencil buffer texture for painting  
![Stencil Texture](docs/images/stencil_buffer_example.png)

UDIM material layout  
![UDIM Example](docs/images/udim_example.png)

## Usage Workflow

1. Extract video frames using the FFmpeg operator
2. Reconstruct the scene using COLMAP
3. Import COLMAP results into Blender using the addon
4. Align a base mesh to the point cloud and image planes
5. Sculpt the base mesh for accuracy
6. Create UDIM materials
7. Generate stencil textures to texture paint a detailed model

## Installation

1. Download or clone this repository
2. In Blender, go to Edit > Preferences > Add-ons > Install
3. Select the zip file or source folder, then enable the addon

## Requirements

- Blender 4.3.2 or higher
- COLMAP - you need to specify the full path to COLMAP.bat
- FFmpeg - full path to ffmpeg.exe

## TODO

- [ ] Convert to addon.
- [ ] Add a simple to follow usage tutorial.


## License

MIT License, see LICENSE file for details.

## Credits

Created by me ([GitHub](https://github.com/z80)) to ease 3D reconstruction workflows in Blender for non-artists.
