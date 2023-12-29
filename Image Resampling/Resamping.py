import cv2
import numpy as np

def nearest_neighbor(image, scale):
    height, width, x = image.shape
    # new hieght and width of the image
    new_height=int(height*scale)
    new_width=int(width*scale)
    print(new_height,new_width)
    
    # create a blank image
    new_image = np.zeros((new_height, new_width, 3), dtype=np.uint8)

    h_scale = height / new_height
    w_scale = width / new_width
    for i in range(new_height):
        for j in range(new_width):
          # int- round off to get neighbor
            new_image[i, j] = image[int(i * h_scale), int(j * w_scale)]

    # output the image
    cv2.imwrite('nearest_neighbor_image.jpg', new_image)

def bilinear(image, scale):
    # new image height and width
    height, width, x = image.shape
    new_height=int(height*scale)
    new_width=int(width*scale)
    print(new_height,new_width)

    # create a blank image
    new_image = np.zeros((new_height, new_width, 3), dtype=np.uint8)

    #scale to resample
    h_scale = height / new_height
    w_scale = width / new_width

    for x in range(new_height):
        for y in range(new_width):
            h = x * h_scale
            w = y * w_scale
            i=int(h)
            j=int(w)
            a= h-i
            b= w-j
            i2, j2 = min(i + 1, height - 1), min(j + 1, width - 1)

            #F(x,y)= (1-a)(1-b)F(i,j) + 
            #        (1-a)b F(i,j+1)+
            #        a(1-b) F(i+1,j)+
            #        ab F(i+1,j+1) 

            # bottom_left=F(i,j)   bottom_right=F(i+1,j) top_left=F(i,j+1)

            interpolated_pixel = [
                int(
                    (1 - a) * (1 - b) * image[i, j][0] +
                    (1 - a) * b * image[i, j2][0] +
                    a * (1 - b) * image[i2, j][0] +
                    a * b * image[i2, j2][0]
                ),
                int(
                    (1 - a) * (1 - b) * image[i, j][1] +
                    (1 - a) * b * image[i, j2][1] +
                    a * (1 - b) * image[i2, j][1] +
                    a * b * image[i2, j2][1]
                ),
                int(
                    (1 - a) * (1 - b) * image[i, j][2] +
                    (1 - a) * b * image[i, j2][2] +
                    a * (1 - b) * image[i2, j][2] +
                    a * b * image[i2, j2][2]
                )
            ]

            new_image[x][y] = interpolated_pixel

    cv2.imwrite('bilinear.jpg',new_image)

#read the image
image = cv2.imread('/content/download.jpg')
height, width, x = image.shape
print(height, width)

scale = 1.5
nearest_neighbor_image = nearest_neighbor(image, scale)
biLinear_image= bilinear(image, scale)
