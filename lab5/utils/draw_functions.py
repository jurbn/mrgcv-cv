import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv

def plot2Ddistances(i,image,x1,x1_p,title):
    plt.figure(i)
    plt.imshow(image, cmap="gray", vmin=0, vmax=255)
    plotResidual(x1, x1_p, "k-")
    plt.plot(x1_p[0, :], x1_p[1, :], "bo")
    plt.plot(x1[0, :], x1[1, :], "rx")
    plotNumberedImagePoints(x1[0:2, :], "r", 4)
    plt.title(title)
    plt.plot()
    plt.show()
    return 0

def EpipolarLine(image,l):
    img = cv.cvtColor(cv.imread(image), cv.COLOR_BGR2RGB)
    # line equation  => a*x + b*y + c = 0
        # p_l_y is the intersection of the line with the axis Y (x=0)
    p_l_y = np.array([0, -l[2] /  l[1]])
    # p_l_x is the intersection point of the line with the axis X (y=0)
    p_l_x = np.array([-l[2] /l[0], 0])

    print([p_l_y[1], p_l_x[1]])
    print([p_l_y[0], p_l_x[0]])
    if(p_l_y[1]<0):
        p_l_y[1] = -(l[0]*750+l[2])/l[1] 
        p_l_y[0] = 750
      

    plt.figure(1)
    plt.figure(figsize=(5,5))
    plt.imshow(img, cmap='gray', vmin=0, vmax=255)
    plt.title(image)
    plt.plot([p_l_y[0], p_l_x[0]], [p_l_y[1], p_l_x[1]], '-r', linewidth=3)
    plt.draw()  # We update the figure display
    print('Click in the image to continue...')
    plt.waitforbuttonpress()
    return 0

def clickPoint(image):
    img = cv.cvtColor(cv.imread(image), cv.COLOR_BGR2RGB)
    plt.figure(2)
    plt.imshow(img)
    plt.title(format(image) + ' - Click a point')
    coord_clicked_point = plt.ginput(1, show_clicks=False)
    p_clicked = np.array([coord_clicked_point[0][0], coord_clicked_point[0][1]])
    plt.plot(p_clicked[0], p_clicked[1], '+r', markersize=15)
    plt.text(p_clicked[0], p_clicked[1],
             "Point {0:.2f}, {1:.2f}".format(p_clicked[0], p_clicked[1]),
             fontsize=15, color='r')
    plt.draw()  # We update the former image without create a new context
    plt.waitforbuttonpress()
    return p_clicked

def draw3DLine(ax, xIni, xEnd, strStyle, lColor, lWidth):
    """
    Draw a segment in a 3D plot
    -input:
        ax: axis handle
        xIni: Initial 3D point.
        xEnd: Final 3D point.
        strStyle: Line style.
        lColor: Line color.
        lWidth: Line width.
    """
    ax.plot([np.squeeze(xIni[0]), np.squeeze(xEnd[0])], [np.squeeze(xIni[1]), np.squeeze(xEnd[1])], [np.squeeze(xIni[2]), np.squeeze(xEnd[2])],
            strStyle, color=lColor, linewidth=lWidth)

def drawRefSystem(ax, T_w_c, strStyle, nameStr):
    """
        Draw a reference system in a 3D plot: Red for X axis, Green for Y axis, and Blue for Z axis
    -input:
        ax: axis handle
        T_w_c: (4x4 matrix) Reference system C seen from W.
        strStyle: lines style.
        nameStr: Name of the reference system.
    """
    draw3DLine(ax, T_w_c[0:3, 3:4], T_w_c[0:3, 3:4] + T_w_c[0:3, 0:1], strStyle, 'r', 1)
    draw3DLine(ax, T_w_c[0:3, 3:4], T_w_c[0:3, 3:4] + T_w_c[0:3, 1:2], strStyle, 'g', 1)
    draw3DLine(ax, T_w_c[0:3, 3:4], T_w_c[0:3, 3:4] + T_w_c[0:3, 2:3], strStyle, 'b', 1)
    ax.text(np.squeeze( T_w_c[0, 3]+0.1), np.squeeze( T_w_c[1, 3]+0.1), np.squeeze( T_w_c[2, 3]+0.1), nameStr)

def plotNumbered3DPoints(ax, X,strColor, offset):
    """
        Plot indexes of points on a 3D plot.
         -input:
             ax: axis handle
             X: Points coordinates.
             strColor: Color of the text.
             offset: Offset from the point to the text.
         -output: None
         """
    for k in range(X.shape[1]):
        ax.text(X[0, k]+offset[0], X[1, k]+offset[1], X[2,k]+offset[2], str(k), color=strColor)

def plotResidual(x,xProjected,strStyle):
    """
        Plot the residual between an image point and an estimation based on a projection model.
         -input:
             x: Image points.
             xProjected: Projected points.
             strStyle: Line style.
         -output: None
         """

    for k in range(x.shape[1]):
        plt.plot([x[0, k], xProjected[0, k]], [x[1, k], xProjected[1, k]], strStyle)

def plotNumberedImagePoints(x,strColor,offset):
    """
        Plot indexes of points on a 2D image.
         -input:
             x: Points coordinates.
             strColor: Color of the text.
             offset: Offset from the point to the text.
         -output: None
         """
    for k in range(x.shape[1]):
        plt.text(x[0, k]+offset, x[1, k]+offset, str(k), color=strColor)