import numpy as np

def cls_type_to_id(cls_type):
    type_to_id={'Car':1,'Pedestrain':2,'Cyclist':3,'Van':4}
    if cls_type not in type_to_id.keys():
        return -1
    return type_to_id[cls_type]

class Object3d(object):
    def __init__(self,line):
        label=line.strip().split(' ')
        self.src=line
        self.cls_type=label[0]
        self.cls_id=cls_type_to_id(self.cls_type)
        self.trucation=float(label[1])
        self.occlusion=float(label[2])
        self.alpha=float(label[3])
        self.box2d=np.array((float(label[4]),float(label[5]),float(label[6]),float(label[7])),dtype=np.float32)
        self.h=float(label[8])
        self.w=float(label[9])
        self.l=float(label[10])
        self.pos=np.array((float(label[11]),float(label[12]),float(label[13])),dtype=np.float32)
        self.dis_to_cam=np.linalg.norm(self.pos)
        self.ry=float(label[14])
        self.score=float(label[15]) if label.__len__()==16 else -1.0
        self.level_str=None
        self.level=self.get_obj_level()

    def get_obj_level(self):
        height=float(self.box2d[3])-float(self.box2d[1])+1

        if height>=40 and self.trucation<=0.15 and self.occlusion<=0:
            self.level_str='Easy'
            return 1
        elif height>=25 and self.trucation<=0.3 and self.occlusion<=1:
            self.level_str='Moderate'
            return 2
        elif height>=25 and self.trucation<=0.5 and self.occlusion<=2:
            self.level_str='Hard'
            return 3
        else:
            self.level_str='Unknown'
            return 4

    def generate_corner3d(self):
        '''
        generate corner3d representation for this object
        :return corner3d:(8,3) corners of box3d in camera coord
        '''
        l,h,w=self.l,self.h,self.w
        x_corners=[l/2,l/2,-l/2,-l/2,l/2,l/2,-l/2,-l/2]
        y_corners=[0,0,0,0,-h,-h,-h,-h]
        z_corners=[w/2,-w/2,-w/2,w/2,w/2,-w/2,-w/2,w/2]

        R=np.array([
            [np.cos(self.ry),0,np.sin(self.ry)],
            [0,1,0],
            [-np.sin(self.ry),0,np.cos(self.ry)]])

        corners3d=np.vstack([x_corners,y_corners,z_corners])
        corners3d=np.dot(R,corners3d).T
        corners3d=corners3d+self.pos
        return corners3d

