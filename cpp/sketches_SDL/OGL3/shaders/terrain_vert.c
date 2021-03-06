#version 330 core

//http://www.opengl-tutorial.org/beginners-tutorials/tutorial-8-basic-shading/

// IN --- Input vertex data, different for all executions of this shader.
layout(location = 0) in vec3 vertPos_model;

// OUT --- Output data ; will be interpolated for each fragment.
noperspective out vec3 fragPos_world;

// UNI --- Values that stay constant for the whole mesh.
uniform vec3 modelPos;
uniform mat4 camMat;

void main(){
	vec3 position_world = modelPos + vertPos_model;
	gl_Position         = camMat   * vec4( position_world, 1 );
	fragPos_world       = position_world;
}



