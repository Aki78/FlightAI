
#include <SDL2/SDL.h>
#include <SDL2/SDL_opengl.h>

#include "AppSDL2OGL.h" // THE HEADER

void AppSDL2OGL::quit(){ SDL_Quit(); loopEnd=true; /*exit(1);*/ };

void AppSDL2OGL::wait(int ms){
    for( int i=0; i<ms; i+=timeSlice ){
        uint32_t tnow=SDL_GetTicks();
        if(tnow>=(upTime+ms)){upTime=tnow; break; }
        SDL_Delay(timeSlice);
    }
};


void AppSDL2OGL::loop( int n ){
	loopEnd = false;
	for( int iframe=0; iframe<n; iframe++ ){
		inputHanding();
		//if(!STOP){update();} // DEPRECATED: usually we want to stop physics, not drawing
		update();
		//printf(" %i \n", iframe );
        wait(delay);
		//if( delay>0 ) SDL_Delay( delay );
		frameCount++;
		if(loopEnd) break;
	}
}

void AppSDL2OGL::inputHanding(){
    const Uint8 *keys = SDL_GetKeyboardState(NULL);
	keyStateHandling ( keys );
	mouseHandling( );
    SDL_Event		 event;
	while(SDL_PollEvent(&event)){
	    eventHandling( event );
	}
}

void AppSDL2OGL::eventHandling( const SDL_Event& event ){
    switch( event.type ){
        case SDL_MOUSEBUTTONDOWN:
            switch( event.button.button ){
                case SDL_BUTTON_LEFT:  LMB = true;  break;
                case SDL_BUTTON_RIGHT: RMB = true;  break;
            };  break;
        case SDL_MOUSEBUTTONUP:
            switch( event.button.button ){
                case SDL_BUTTON_LEFT:  LMB = false; break;
                case SDL_BUTTON_RIGHT: RMB = false; break;
            }; break;
        case SDL_KEYDOWN :
            switch( event.key.keysym.sym ){
                case SDLK_ESCAPE:   quit(); break;
                //case SDLK_SPACE:    STOP = !STOP; printf( STOP ? " STOPED\n" : " UNSTOPED\n"); break;
                case SDLK_KP_MINUS: zoom*=VIEW_ZOOM_STEP; break;
                case SDLK_KP_PLUS:  zoom/=VIEW_ZOOM_STEP; break;
            } break;
        case SDL_QUIT: quit(); break;
    };
};

void AppSDL2OGL::keyStateHandling( const Uint8 *keys ){
    if( keys[ SDL_SCANCODE_LEFT  ] ){ camX0 -= camStep; }
	if( keys[ SDL_SCANCODE_RIGHT ] ){ camX0 += camStep; }
	if( keys[ SDL_SCANCODE_UP    ] ){ camY0 += camStep; }
	if( keys[ SDL_SCANCODE_DOWN  ] ){ camY0 -= camStep; }
};

void AppSDL2OGL::mouseHandling( ){
    SDL_GetMouseState( &mouseX, &mouseY ); //mouseY=HEIGHT-mouseY; // this is done in mouseUp()
    defaultMouseHandling( mouseX, mouseY );
}

void AppSDL2OGL::defaultMouseHandling( const int& mouseX, const int& mouseY ){
	mouse_begin_x = mouseRight( mouseX ) + camX0;
	mouse_begin_y = mouseUp   ( mouseY ) + camY0;
    fWIDTH  = zoom*ASPECT_RATIO;
	fHEIGHT = zoom;
	camXmin = camX0 - fWIDTH; camYmin = camY0 - fHEIGHT;
	camXmax = camX0 + fWIDTH; camYmax = camY0 + fHEIGHT;
};

AppSDL2OGL::AppSDL2OGL( int& id, int WIDTH_, int HEIGHT_ ) : ScreenSDL2OGL( id, WIDTH_, HEIGHT_ ) {

}

