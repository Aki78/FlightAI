
#ifndef Demo_h
#define Demo_h

#include <stdlib.h>
#include <stdio.h>

#include <SDL2/SDL.h>
#include <SDL2/SDL_opengl.h>
#include <GL/glu.h>

#include "fastmath.h"
#include "Vec3.h"
#include "Mat3.h"

// generic interface for any game object exported in plugin (all plugins derive from it)
class Demo{  public:
    virtual void setup()=0;
    virtual void draw()=0;
    virtual void onMouse(float x, float y, uint8_t buttons)=0;
};

// functions exported from plugin
extern "C" {
    void setup();
    void draw();
    void onMouse( float x, float y, uint8_t buttons );
    Demo* CreateDemo();
}

// functions pointer types for functions exported from plugin
typedef void (*Pprocedure)();
typedef void (*Pfunc2f   )(float,float);
typedef void (*PmouseFunc)(float,float,uint8_t);
typedef Demo* (*PDemoFactory)();

#endif

