def Smooth_Closed_Surface_Quadrature_RBF_Test():
    import math
    import numpy as np
    from Smooth_Closed_Surface_Quadrature_RBF import Smooth_Closed_Surface_Quadrature_RBF
    from scipy.spatial import ConvexHull
    import time
    #==========================================================================
    #   
    # This function provides example calls to the function
    # Smooth_Closed_Surface_Quadrature_RBF.py.  The user can change the parameter
    # Number_of_Quadrature_Nodes in this py-file to generate various sizes of
    # node sets.  The user can also change the parameters Poly_Order and
    # Number_of_Nearest_Neighbors within the Smooth_Closed_Surface_Quadrature_RBF.py
    # function.
    #
    # This function approximates the surface integral of five test integrands
    # over the surface of the sphere and prints the error in the approximation
    # to the command windown upon completion.  The sphere is used here since it
    # is a simple matter to generate random node sets for the surface.
    #
    # The function Smooth_Closed_Surface_Quadrature_RBF.py is the default implemenation of
    # the method described in:
    #
    # J. A. Reeger, B. Fornberg, and M. L. Watts "Numerical quadrature over
    # smooth, closed surfaces".
    #
    #==========================================================================

    #==========================================================================
    #
    # Generate quadrature nodes on the surface of a sphere of radius 1 using
    # the method describe in A. Gonzalez, "Measurement of Areas on a Sphere
    # using Fibonacci and Latitude-Longitude Lattices", Math. Geosci., vol. 42,
    # number 1, pages 49-64, 2010
    #
    #==========================================================================
    
    # Parameters user can set
    Number_of_Quadrature_Nodes=2500;
    
    # Generate the quadrature nodes
    phi=(1+np.sqrt(5))/2;
    
    Quadrature_Nodes=np.zeros((Number_of_Quadrature_Nodes,3));
    
    N=Number_of_Quadrature_Nodes/2;
    
    Sphere_Radius=1;
    
    for Index in range(-N,N):
        lat=np.arcsin((2.*Index)/(2.*N+1))
        lon=(Index % phi)*2*np.pi/phi;
        if lon<-np.pi:
            lon=2*np.pi+lon;
        if lon>np.pi:
            lon=lon-2*np.pi;
        Quadrature_Nodes[Index+N,:]=[np.cos(lon)*np.cos(lat),np.sin(lon)*np.cos(lat),np.sin(lat)];

    # Generate a triangulation of the surface
    hull=ConvexHull(Quadrature_Nodes); 
    Triangles=hull.simplices; 
    #==========================================================================

    #==========================================================================
    #
    # Define the level surface h(x,y,z)=0 and its gradient so that the normal
    # to the surface can be computed exactly.  These are inputs to the function
    # that can be omitted.  They have the following specifications
    #
    #   h (optional) - For the surface S defined implicitly by h(x,y,z)=0, row i
    #   in the output of h should contain
    #   h(Quadrature_Nodes(i,1:3))
    #   h should take in Quadrature_Nodes as an 
    #   (Number_of_Quadrature_Nodes X 3) Array
    #
    #   gradh (optional) - The gradient of the function h.  Row i in the output
    #   of gradh should contain
    #   [dh/dx(Quadrature_Nodes(i,1:3)),dh/dy(Quadrature_Nodes(i,1:3)),dh/dz(Quadrature_Nodes(i,1:3)]
    #   gradh should take in Quadrature_Nodes as an
    #   (Number_of_Quadrature_Nodes X 3) Array
    #
    #==========================================================================
    h=lambda p: p[:,0]*p[:,0]+p[:,1]*p[:,1]+p[:,2]*p[:,2]-Sphere_Radius*Sphere_Radius;
    gradh=lambda p: 2*p;
    #==========================================================================

    #==========================================================================
    #
    # Generate the quadrature weights for the set of quadrature nodes generated
    # above.
    #
    #==========================================================================
    # Use the exact surface normal
    Quadrature_Weights_Exact_Normal=Smooth_Closed_Surface_Quadrature_RBF(Quadrature_Nodes,Triangles,h,gradh);
    
    # Use the approximate surface normal
    Quadrature_Weights_Approx_Normal=Smooth_Closed_Surface_Quadrature_RBF(Quadrature_Nodes,Triangles);
    #==========================================================================
    
    #==========================================================================
    #
    # Define test integrands as lambda functions
    #
    #==========================================================================
    f1=lambda p:1+p[:,0]+np.power(p[:,1],2)+np.power(p[:,0],2)*p[:,1]+np.power(p[:,0],4)+np.power(p[:,1],5)+np.power(p[:,0],2)*np.power(p[:,1],2)*np.power(p[:,2],2);
    f2=lambda p:0.75*np.exp(-np.power((9*p[:,0]-2),2)/4-np.power((9*p[:,1]-2),2)/4-np.power((9*p[:,2]-2),2)/4)+0.75*np.exp(-np.power((9*p[:,0]+1),2)/49-(9*p[:,1]+1)/10-(9*p[:,2]+1)/10)+0.5*np.exp(-np.power((9*p[:,0]-7),2)/4-np.power((9*p[:,1]-3),2)/4-np.power((9*p[:,2]-5),2)/4)-0.2*np.exp(-np.power((9*p[:,0]-4),2)-np.power((9*p[:,1]-7),2)-np.power((9*p[:,2]-5),2));
    f3=lambda p:(1+np.tanh(-9*p[:,0]-9*p[:,1]+9*p[:,2]))/9;
    f4=lambda p:(1+np.sign(-9*p[:,0]-9*p[:,1]+9*p[:,2]))/9;
    f5=lambda p:np.ones(Number_of_Quadrature_Nodes);
    #==========================================================================
        
    #==========================================================================
    #
    # Compute the values of various test integrands at the quadrature nodes
    # generated above.
    #
    #==========================================================================
    F1=f1(Quadrature_Nodes);
    F2=f2(Quadrature_Nodes);
    F3=f3(Quadrature_Nodes);
    F4=f4(Quadrature_Nodes);
    F5=f5(Quadrature_Nodes);
    #==========================================================================

    #==========================================================================
    #
    # Set the exact values of the surface integrals of the test integrands for
    # comparison.
    #
    #==========================================================================
    Exact_Surface_Integral_f1=216*np.pi/35;
    Exact_Surface_Integral_f2=6.6961822200736179523;
    Exact_Surface_Integral_f3=4*np.pi/9;
    Exact_Surface_Integral_f4=4*np.pi/9;
    Exact_Surface_Integral_f5=4*np.pi;
    #==========================================================================

    #==========================================================================
    #
    # Compute the approximate values of the surface integrals of the test
    # integrands using the quadrature weights generated above for comparison.
    #
    #==========================================================================
    Approximate_Surface_Integral_f1_Exact_Normal=np.dot(F1,Quadrature_Weights_Exact_Normal);
    Approximate_Surface_Integral_f2_Exact_Normal=np.dot(F2,Quadrature_Weights_Exact_Normal);
    Approximate_Surface_Integral_f3_Exact_Normal=np.dot(F3,Quadrature_Weights_Exact_Normal);
    Approximate_Surface_Integral_f4_Exact_Normal=np.dot(F4,Quadrature_Weights_Exact_Normal);
    Approximate_Surface_Integral_f5_Exact_Normal=np.dot(F5,Quadrature_Weights_Exact_Normal);
    
    Approximate_Surface_Integral_f1_Approx_Normal=np.dot(F1,Quadrature_Weights_Approx_Normal);
    Approximate_Surface_Integral_f2_Approx_Normal=np.dot(F2,Quadrature_Weights_Approx_Normal);
    Approximate_Surface_Integral_f3_Approx_Normal=np.dot(F3,Quadrature_Weights_Approx_Normal);
    Approximate_Surface_Integral_f4_Approx_Normal=np.dot(F4,Quadrature_Weights_Approx_Normal);
    Approximate_Surface_Integral_f5_Approx_Normal=np.dot(F5,Quadrature_Weights_Approx_Normal);
    #==========================================================================

    #==========================================================================
    #
    # Compute the error in the approximation of the surface integrals for the
    # test integrands.
    #
    #==========================================================================
    Error_in_the_Approximate_Surface_Integral_f1_Exact_Normal=np.abs(Exact_Surface_Integral_f1-
        Approximate_Surface_Integral_f1_Exact_Normal)/np.abs(Approximate_Surface_Integral_f1_Exact_Normal);
    Error_in_the_Approximate_Surface_Integral_f2_Exact_Normal=np.abs(Exact_Surface_Integral_f2-
        Approximate_Surface_Integral_f2_Exact_Normal)/np.abs(Approximate_Surface_Integral_f2_Exact_Normal);
    Error_in_the_Approximate_Surface_Integral_f3_Exact_Normal=np.abs(Exact_Surface_Integral_f3-
        Approximate_Surface_Integral_f3_Exact_Normal)/np.abs(Approximate_Surface_Integral_f3_Exact_Normal);
    Error_in_the_Approximate_Surface_Integral_f4_Exact_Normal=np.abs(Exact_Surface_Integral_f4-
        Approximate_Surface_Integral_f4_Exact_Normal)/np.abs(Approximate_Surface_Integral_f4_Exact_Normal);
    Error_in_the_Approximate_Surface_Integral_f5_Exact_Normal=np.abs(Exact_Surface_Integral_f5-
        Approximate_Surface_Integral_f5_Exact_Normal)/np.abs(Approximate_Surface_Integral_f5_Exact_Normal);
    
    Error_in_the_Approximate_Surface_Integral_f1_Approx_Normal=np.abs(Exact_Surface_Integral_f1-
        Approximate_Surface_Integral_f1_Approx_Normal)/np.abs(Approximate_Surface_Integral_f1_Approx_Normal);
    Error_in_the_Approximate_Surface_Integral_f2_Approx_Normal=np.abs(Exact_Surface_Integral_f2-
        Approximate_Surface_Integral_f2_Approx_Normal)/np.abs(Approximate_Surface_Integral_f2_Approx_Normal);
    Error_in_the_Approximate_Surface_Integral_f3_Approx_Normal=np.abs(Exact_Surface_Integral_f3-
        Approximate_Surface_Integral_f3_Approx_Normal)/np.abs(Approximate_Surface_Integral_f3_Approx_Normal);
    Error_in_the_Approximate_Surface_Integral_f4_Approx_Normal=np.abs(Exact_Surface_Integral_f4-
        Approximate_Surface_Integral_f4_Approx_Normal)/np.abs(Approximate_Surface_Integral_f4_Approx_Normal);
    Error_in_the_Approximate_Surface_Integral_f5_Approx_Normal=np.abs(Exact_Surface_Integral_f5-
        Approximate_Surface_Integral_f5_Approx_Normal)/np.abs(Approximate_Surface_Integral_f5_Approx_Normal);
    #==========================================================================
    
    
    #==========================================================================
    #
    # Print some stuff
    #
    #==========================================================================    
    print '====================================================================' 
    print 'The relative error in the approximation of the surface integral of' 
    print 'f1(x,y,z)= '
    print '         4    2  2  2    2          5    2'
    print '        x  + x  y  z  + x  y + x + y  + y  + 1'
    print 'over the sphere surface (radius 1) with exact normal is'
    print Error_in_the_Approximate_Surface_Integral_f1_Exact_Normal[0]
    
    print '====================================================================' 
    print 'The relative error in the approximation of the surface integral of' 
    print 'f2(x,y,z)=                                                        '
    print '          /                        2     \ '
    print '     3    |   9 y   9 z   (9 x + 1)    1 |'
    print '     - exp| - --- - --- - ---------- - - |'
    print '     4    \    10    10       49       5 /'
    print ''
    print '     1               2            2            2'
    print '   - - exp(- (9 x - 4)  - (9 y - 7)  - (9 z - 5) )'
    print '     5 '
    print ''
    print '          /            2            2            2 \ '
    print '     3    |   (9 x - 2)    (9 y - 2)    (9 z - 2)  |'
    print '   + - exp| - ---------- - ---------- - ---------- |'
    print '     4    \        4            4            4     /'
    print ''
    print '           /            2            2            2 \ '
    print '     1     |   (9 x - 7)    (9 y - 3)    (9 z - 5)  |'
    print '   + -  exp| - ---------- - ---------- - ---------- |'
    print '     2     \        4            4            4     /'
    print 'over the sphere surface (radius 1) with exact normal is'
    print Error_in_the_Approximate_Surface_Integral_f2_Exact_Normal[0]
    
    print '====================================================================' 
    print 'The relative error in the approximation of the surface integral of' 
    print 'f3(x,y,z)=                                                        '
    print '     1   tanh(9 x + 9 y - 9 z)'
    print '   - -   ---------------------'
    print '     9             9          '
    print 'over the sphere surface (radius 1) with exact normal is'
    print Error_in_the_Approximate_Surface_Integral_f3_Exact_Normal[0]
    
    print '====================================================================' 
    print 'The relative error in the approximation of the surface integral of' 
    print 'f4(x,y,z)=                                                        '
    print '     1   sign(9 x + 9 y - 9 z)'
    print '   - -   ---------------------'
    print '     9             9          '
    print 'over the sphere surface (radius 1) with exact normal is'
    print Error_in_the_Approximate_Surface_Integral_f4_Exact_Normal[0]
    
    print '====================================================================' 
    print 'The relative error in the approximation of the surface integral of' 
    print 'f5(x,y,z)=1 over the sphere surface (radius 1) with exact normal is'
    print Error_in_the_Approximate_Surface_Integral_f5_Exact_Normal[0]
    print '====================================================================' 
    
    print '' 
    
    print '====================================================================' 
    print 'The relative error in the approximation of the surface integral of' 
    print 'f1(x,y,z)= '
    print '         4    2  2  2    2          5    2'
    print '        x  + x  y  z  + x  y + x + y  + y  + 1'
    print 'over the sphere surface (radius 1) with approximate normal is'
    print Error_in_the_Approximate_Surface_Integral_f1_Approx_Normal[0]
    
    print '====================================================================' 
    print 'The relative error in the approximation of the surface integral of' 
    print 'f2(x,y,z)=                                                        '
    print '          /                        2     \ '
    print '     3    |   9 y   9 z   (9 x + 1)    1 |'
    print '     - exp| - --- - --- - ---------- - - |'
    print '     4    \    10    10       49       5 /'
    print ''
    print '     1               2            2            2'
    print '   - - exp(- (9 x - 4)  - (9 y - 7)  - (9 z - 5) )'
    print '     5 '
    print ''
    print '          /            2            2            2 \ '
    print '     3    |   (9 x - 2)    (9 y - 2)    (9 z - 2)  |'
    print '   + - exp| - ---------- - ---------- - ---------- |'
    print '     4    \        4            4            4     /'
    print ''
    print '           /            2            2            2 \ '
    print '     1     |   (9 x - 7)    (9 y - 3)    (9 z - 5)  |'
    print '   + -  exp| - ---------- - ---------- - ---------- |'
    print '     2     \        4            4            4     /'
    print 'over the sphere surface (radius 1) with approximate normal is'
    print Error_in_the_Approximate_Surface_Integral_f2_Approx_Normal[0]
    
    print '====================================================================' 
    print 'The relative error in the approximation of the surface integral of' 
    print 'f3(x,y,z)=                                                        '
    print '     1   tanh(9 x + 9 y - 9 z)'
    print '   - -   ---------------------'
    print '     9             9          '
    print 'over the sphere surface (radius 1) with approximate normal is'
    print Error_in_the_Approximate_Surface_Integral_f3_Approx_Normal[0]
    
    print '====================================================================' 
    print 'The relative error in the approximation of the surface integral of' 
    print 'f4(x,y,z)=                                                        '
    print '     1   sign(9 x + 9 y - 9 z)'
    print '   - -   ---------------------'
    print '     9             9          '
    print 'over the sphere surface (radius 1) with approximate normal is'
    print Error_in_the_Approximate_Surface_Integral_f4_Approx_Normal[0]
    
    print '====================================================================' 
    print 'The relative error in the approximation of the surface integral of' 
    print 'f5(x,y,z)=1 over the sphere surface (radius 1) with approximate normal is'
    print Error_in_the_Approximate_Surface_Integral_f5_Approx_Normal[0]
    print '====================================================================' 
    #==========================================================================    
    
    return