# import numpy as np
# a = [[1,2,3],
#      [4,5,6],
#      [7,8,9],
#      [10,11,12]]
# print(a)
# print(a[np.arange(2), 2])
#
#
def doit(cube1,n):
    nom = n + 1
    if(nom >= 9):
        return
    cuberd = rd(cube)
    doit(cuberd,nom)

    cuberup = rup(cube)
    doit(cuberup,nom)
    cubeld = ld(cube)
    doit(cubeld,nom)
    cubelup = lup(cube)
    doit(cubelup,nom)
    cubeur = ur(cube)
    doit(cubeur,nom)
    cubeul = ul(cube)
    doit(cubeul,nom)
    cubedr = dr(cube)
    doit(cubedr,nom)
    cubedl = dl(cube)
    doit(cubedl,nom)
    cubefc =fc(cube)
    doit(cubefc,nom)
    cubefoc = foc(cube)
    doit(cubefoc,nom)
    cubebc = bc(cube)
    doit(cubebc,nom)
    cubeboc = boc(cube)
    doit(cubeboc,nom)
