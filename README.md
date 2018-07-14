# Structred-Light
Reconstruct a scene from multiple structured light scannings of it.
Steps:-
1. Calibrate projector with the “easy” method
    Use ray-plane intersection
    Get 2D-3D correspondence and use stereo calibration
2. Get the binary code for each pixel.
3. Correlate code with (x,y) position provided in "codebook" from binary code -> (x,y)
4. With 2D-2D correspondence,p erform stereo triangulation (existing function) to get a depth map
5. Add color to your 3D cloud
6. When finding correspondences, take the RGB values from "aligned001.png"
7. Add them later to your reconstruction
8. Output is given in a file called "output_color.xyzrgb" with the following format
    "%d %d %d %d %d %d\n"%(x, y, z, r, g, b)
    for each 3D+RGB poin
