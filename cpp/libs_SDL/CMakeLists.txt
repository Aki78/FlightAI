find_package( OpenGL REQUIRED )
find_package( GLU    REQUIRED )
find_package( SDL2   REQUIRED )

project(libs_SDL)

# problem is that these libraries cannot be build from common objects .o; needs "-fPIC" to relocation
#add_subdirectory (libSail)
#add_subdirectory (Molecular)
add_subdirectory (Vis3D)
add_subdirectory (FlightView)
#add_subdirectory (Shock)

