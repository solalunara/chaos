import { initBuffers } from "./init-buffers.js";
import { drawScene } from "./draw-scene.js";

// Vertex shader program
const vsSource = `
    attribute vec4 aVertexPosition;
    attribute vec4 aVertexColor;

    uniform mat4 uModelViewMatrix;
    uniform mat4 uProjectionMatrix;

    varying lowp vec4 vColor;

    void main(void) {
      gl_Position = uProjectionMatrix * uModelViewMatrix * aVertexPosition;
      vColor = aVertexColor;
    }
  `;

// Fragment shader program
const fsSource = `
    varying lowp vec4 vColor;

    void main(void) {
      gl_FragColor = vColor;
    }
  `;


function initShaderProgram(gl, vsSource, fsSource) {
    const vertexShader = loadShader(gl, gl.VERTEX_SHADER, vsSource);
    const fragmentShader = loadShader(gl, gl.FRAGMENT_SHADER, fsSource);

    // Create the shader program

    const shaderProgram = gl.createProgram();
    gl.attachShader(shaderProgram, vertexShader);
    gl.attachShader(shaderProgram, fragmentShader);
    gl.linkProgram(shaderProgram);

    // If creating the shader program failed, alert

    if (!gl.getProgramParameter(shaderProgram, gl.LINK_STATUS)) {
        alert(
        `Unable to initialize the shader program: ${gl.getProgramInfoLog(
            shaderProgram,
        )}`,
        );
        return null;
    }

    return shaderProgram;
}

//
// creates a shader of the given type, uploads the source and
// compiles it.
//
function loadShader(gl, type, source) {
    const shader = gl.createShader(type);

    // Send the source to the shader object

    gl.shaderSource(shader, source);

    // Compile the shader program

    gl.compileShader(shader);

    // See if it compiled successfully

    if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
        alert(
        `An error occurred compiling the shaders: ${gl.getShaderInfoLog(shader)}`,
        );
        gl.deleteShader(shader);
        return null;
    }

    return shader;
}

const phaseDiv = document.querySelector( "#phasespace" );
const realDiv = document.querySelector( "#realspace" );


const phaseCanvas = document.querySelector( "#phaseCanvas" );
const realCanvas = document.querySelector( "#realCanvas" );
const gl_phase = phaseCanvas.getContext( "webgl" );
const gl_real = realCanvas.getContext( "webgl" );

phaseCanvas.width = phaseDiv.getBoundingClientRect().width;
phaseCanvas.height = phaseDiv.getBoundingClientRect().height;
realCanvas.width = realDiv.getBoundingClientRect().width;
realCanvas.height = realDiv.getBoundingClientRect().height;

if ( gl_phase === null || gl_real === null ) {
    alert( "Unable to initialize WebGL. Your browser or machine may not support it." );
}

// Set clear color to black, fully opaque
gl_phase.clearColor(0.0, 0.0, 0.0, 1.0);
gl_real.clearColor(0.0, 0.0, 0.0, 1.0);
// Clear the color buffer with specified clear color
gl_phase.clear(gl_phase.COLOR_BUFFER_BIT);
gl_real.clear(gl_real.COLOR_BUFFER_BIT);

const phaseShaderProgram = initShaderProgram(gl_phase, vsSource, fsSource);
const realShaderProgram = initShaderProgram(gl_real, vsSource, fsSource);

const phaseProgramInfo = {
    program: phaseShaderProgram,
    attribLocations: {
        vertexPosition: gl_phase.getAttribLocation(phaseShaderProgram, "aVertexPosition"),
        vertexColor: gl_phase.getAttribLocation(phaseShaderProgram, "aVertexColor"),
    },
    uniformLocations: {
        projectionMatrix: gl_phase.getUniformLocation(phaseShaderProgram, "uProjectionMatrix"),
        modelViewMatrix: gl_phase.getUniformLocation(phaseShaderProgram, "uModelViewMatrix"),
    },
};
const realProgramInfo = {
    program: realShaderProgram,
    attribLocations: {
        vertexPosition: gl_real.getAttribLocation(realShaderProgram, "aVertexPosition"),
        vertexColor: gl_real.getAttribLocation(realShaderProgram, "aVertexColor"),
    },
    uniformLocations: {
        projectionMatrix: gl_real.getUniformLocation(realShaderProgram, "uProjectionMatrix"),
        modelViewMatrix: gl_real.getUniformLocation(realShaderProgram, "uModelViewMatrix"),
    },
};

// Here's where we call the routine that builds all the
// objects we'll be drawing.
const buffers_phase = initBuffers(gl_phase);
const buffers_real = initBuffers(gl_real);

// Draw the scene
drawScene(gl_phase, phaseProgramInfo, buffers_phase);
drawScene(gl_real, realProgramInfo, buffers_real);
