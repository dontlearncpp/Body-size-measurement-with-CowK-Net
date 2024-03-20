import cv2
import numpy as np
import open3d as o3d
import math
import time
import csv
import os

def neighbors(matrix, rowNumber, colNumber):
    result = []
    for rowAdd in range(-10, 10):
        newRow = rowNumber + rowAdd
        if newRow >= 0 and newRow <= len(matrix)-1:
            for colAdd in range(-10, 10):
                newCol = colNumber + colAdd
                if newCol >= 0 and newCol <= len(matrix)-1:
                    if newCol == colNumber and newRow == rowNumber:
                        continue
                    result.append(matrix[newCol][newRow])
    val = np.array(result)

    # val = [x for x in val if x != 0]
    # result = np.median(val)

    result = val.sum(0) / (val != 0).sum(0)
    return result

def shortest_distance(x1, y1, z1, a, b, c, d):
    d = abs((a * x1 + b * y1 + c * z1 + d))
    e = (math.sqrt(a * a + b * b + c * c))
    print("Perpendicular distance is", d / e)
    return d/e



class point_cloud_generator():

    def __init__(self, rgb_file, depth_file, save_ply, camera_intrinsics=[784.0, 779.0, 649.0, 405.0]):
        self.rgb_file = rgb_file
        self.depth_file = depth_file
        self.save_ply = save_ply
        self.p1 = p1
        self.p2 = p2
        self.p3 = p3
        self.p4 = p4
        self.p5 = p5
        self.p6 = p6



        self.rgb = cv2.imread(rgb_file)
        self.depth = cv2.imread(self.depth_file, -1)
        # self.depth = self.depth[:,:,0]
        print("your depth image shape is:", self.depth.shape)

        self.width = self.rgb.shape[1]
        self.height = self.rgb.shape[0]

        self.camera_intrinsics = camera_intrinsics
        self.depth_scale = 1000

    def compute(self):
        t1 = time.time()

        depth = np.asarray(self.depth, dtype=np.uint16).T
        # depth[depth==65535]=0
        self.Z = depth / self.depth_scale
        fx, fy, cx, cy = self.camera_intrinsics #cx,cy摄像头光学中心，fx,fy摄像头焦距

        X = np.zeros((self.width, self.height))
        Y = np.zeros((self.width, self.height))
        for i in range(self.width):
            X[i, :] = np.full(X.shape[1], i)

        self.X = ((X - cx) * self.Z) / fx
        for i in range(self.height):
            Y[:, i] = np.full(Y.shape[0], i)
        self.Y = ((Y - cy) * self.Z) / fy

        data_ply = np.zeros((6, self.width * self.height))
        data_ply[0] = self.X.T.reshape(-1)
        data_ply[1] = -self.Y.T.reshape(-1)
        data_ply[2] = -self.Z.T.reshape(-1)
        img = np.array(self.rgb, dtype=np.uint8)
        data_ply[3] = img[:, :, 0:1].reshape(-1)
        data_ply[4] = img[:, :, 1:2].reshape(-1)
        data_ply[5] = img[:, :, 2:3].reshape(-1)

        # p1_index = self.p1[0]*self.p1[1]
        # p2_index = self.p2[0]*self.p2[1]
        # p3_index = self.p3[0]*self.p3[1]
        # p4_index = self.p4[0]*self.p4[1]


        P1 = [neighbors(self.X,p1[1],p1[0]),neighbors(self.Y,p1[1],p1[0]),neighbors(self.Z,p1[1],p1[0])]
        P2 = [neighbors(self.X,p2[1],p2[0]),neighbors(self.Y,p2[1],p2[0]),neighbors(self.Z,p2[1],p2[0])]
        P3 = [neighbors(self.X,p3[1],p3[0]),neighbors(self.Y,p3[1],p3[0]),neighbors(self.Z,p3[1],p3[0])]
        P4 = [neighbors(self.X,p4[1],p4[0]),neighbors(self.Y,p4[1],p4[0]),neighbors(self.Z,p4[1],p4[0])]

        P5 = [neighbors(self.X,p5[1],p5[0]),neighbors(self.Y,p5[1],p5[0]),neighbors(self.Z,p5[1],p5[0])]
        P6 = [neighbors(self.X,p6[1],p6[0]),neighbors(self.Y,p6[1],p6[0]),neighbors(self.Z,p6[1],p6[0])]




        self.data_ply = data_ply
        float_formatter = lambda x: "%.4f" % x
        points = []
        for i in self.data_ply.T:
            points.append("{} {} {} {} {} {} 0\n".format
                          (float_formatter(i[0]), float_formatter(i[1]), float_formatter(i[2]),
                           int(i[3]), int(i[4]), int(i[5])))

        file = open(self.save_ply, "w")
        file.write('''ply
        format ascii 1.0
        element vertex %d
        property float x
        property float y
        property float z
        property uchar red
        property uchar green
        property uchar blue
        property uchar alpha
        end_header
        %s
        ''' % (len(points), "".join(points)))
        file.close()

        pcd = o3d.io.read_point_cloud(self.save_ply)
        plane_model, inliers = pcd.segment_plane(distance_threshold=0.03,
                                                 ransac_n=3,
                                                 num_iterations=1000000000)
        [a, b, c, d] = plane_model
        plane_cloud = pcd.select_by_index(inliers)
        plane_cloud.paint_uniform_color([1.0, 0, 0])
        o3d.io.write_point_cloud("plane_cloud.ply", plane_cloud)
        noneplane_cloud = pcd.select_by_index(inliers, invert=True)
        o3d.io.write_point_cloud("noneplane_cloud.ply", noneplane_cloud)



        # o3d.visualization.draw_geometries([plane_cloud, noneplane_cloud,pcd2])

        # -0.05122340266133925,-0.46287818967055727,3.568
        # 0.7884406532053798,-0.44016396420030707,3.365125
        x = P1[0]
        y = P1[1]
        z = P1[2]

        x1 = P3[0]
        y1 = P3[1]
        z1 = P3[2]

        print(f"Plane equation: {a:.2f}x + {b:.2f}y + {c:.2f}z + {d:.2f} = 0")
        tigao = shortest_distance(x, -y, -z, a, b, c, d)
        tungao = shortest_distance(x1, -y1, -z1, a, b, c, d)
        print(shortest_distance(x, y, z, a, b, c, d))
        print(shortest_distance(x1, y1, z1, a, b, c, d))

        with open(r'L:\boshi\dianyunfenge\test\tuxiangfenge\1.csv','a',encoding='utf-8',newline='') as f:
            writer = csv.writer(f)
            img_name = os.path.split(self.rgb_file)[1]
            writer.writerow([img_name])

            for i in (P1,P2,P3,P4,P5,P6):
                csvwriter = csv.writer(f, delimiter=' ')
                csvwriter.writerow([i[0], -i[1],-i[2]])


            temp = np.sqrt((P2[0] - P4[0])**2 + (P2[1] - P4[1])**2 + (P2[2] - P4[2])**2)
            xiongshen = np.sqrt((P5[0] - P6[0])**2 + (P5[1] - P6[1])**2 + (P5[2] - P6[2])**2)
            writer.writerow(['tixiechang: '+str(round(temp*100, 1))])
            writer.writerow(['tigao: '+str(round(tigao*100,1))])

            writer.writerow(['tungao: '+str(round(tungao*100,1))])
            writer.writerow(['xiongshen: '+str(round(xiongshen*100,1))])


        t2 = time.time()
        print('calcualte 3d point cloud Done.', t2 - t1)



    def write_ply(self):
        start = time.time()
        float_formatter = lambda x: "%.4f" % x
        points = []
        for i in self.data_ply.T:
            points.append("{} {} {} {} {} {} 0\n".format
                          (float_formatter(i[0]), float_formatter(i[1]), float_formatter(i[2]),
                           int(i[3]), int(i[4]), int(i[5])))

        file = open(self.save_ply, "w")
        file.write('''ply
        format ascii 1.0
        element vertex %d
        property float x
        property float y
        property float z
        property uchar red
        property uchar green
        property uchar blue
        property uchar alpha
        end_header
        %s
        ''' % (len(points), "".join(points)))
        file.close()

        end = time.time()
        print("Write into .ply file Done.", end - start)



    def show_point_cloud(self):
        pcd = o3d.io.read_point_cloud(self.save_ply)
        o3d.visualization.draw([pcd])

# [545.77062327 447.78391054 788.13736552 915.16029316 586.18184489
#  585.0772894 ]
# [236.23842001 401.08622611 243.03907156 261.15674525 231.97643459
#  456.97331429]



if __name__ == '__main__':
    # camera_intrinsics = [936.62, 578.234, 1401.55, 1400.85]#fx, fy, cx, cy
    camera_intrinsics = [ 1401.55, 1400.85,936.62, 578.234]#fx, fy, cx, cy
    # camera_intrinsics = [ 1401.55, 1400.85,960, 540]#fx, fy, cx, cy
    xxx=[ 617,537,855,998,674,687 ]
    yyy=[ 222,387,225,257,216,443  ]
    tihua = str('093')


    p1=[xxx[0]  , yyy[0]          ]
    p2=[xxx[1]  , yyy[1]            ]
    p3=[xxx[2]   ,yyy[2]          ]
    p4=[xxx[3]   ,yyy[3]           ]
    p5 = [ xxx[4], yyy[4]]
    p6 = [ xxx[5], yyy[5]]

    p1[0] = int(p1[0]/1333*1902)
    p1[1] = int(p1[1]/750*1080)
    p2[0] = int(p2[0]/1333*1902)
    p2[1] = int(p2[1]/750*1080)
    p3[0] = int(p3[0]/1333*1902)
    p3[1] = int(p3[1]/750*1080)
    p4[0] = int(p4[0]/1333*1902)
    p4[1] = int(p4[1]/750*1080)
    p5[1] = int(p5[1]/750*1080)
    p5[0] = int(p5[0]/1333*1902)
    p6[0] = int(p6[0]/1333*1902)
    p6[1] = int(p6[1]/750*1080)





#1076 866 229

    depth_file = "E:\\guanjindianjiance\\tichi\\depth3\\"+"V"+tihua+".png"
    rgb_file = "E:\\guanjindianjiance\\tichi\\img3\\"+tihua+".jpg"
    save_ply = tihua+".ply"

    wimg = cv2.imread(rgb_file)
    img = cv2.imread(rgb_file)

    cv2.circle(img,p1,20,(0,0,255),-1)
    cv2.circle(img,p2,20,(0,0,255),-1)
    cv2.circle(img,p3,20,(0,0,255),-1)
    cv2.circle(img,p4,20,(0,0,255),-1)
    cv2.circle(img,p5,20,(0,0,255),-1)
    cv2.circle(img,p6,20,(0,0,255),-1)

    cv2.circle(wimg,p1,20,(0,0,255),-1)
    cv2.circle(wimg,p2,20,(0,0,255),-1)
    cv2.circle(wimg,p3,20,(0,0,255),-1)
    cv2.circle(wimg,p4,20,(0,0,255),-1)

 #   cv2.imwrite(r"C:\Users\yang\Documents\Adobe\Premiere Pro\12.0\4"+'\\'+tihua+'.jpg', wimg)
 #   cv2.imwrite(r"C:\Users\yang\Documents\Adobe\Premiere Pro\12.0\3"+'\\'+tihua+'.jpg', img)
    a = point_cloud_generator(rgb_file=rgb_file,
                              depth_file=depth_file,
                              save_ply=save_ply,
                              camera_intrinsics=camera_intrinsics
                              )
    a.compute()
    # a.write_ply()
    # a.show_point_cloud()



