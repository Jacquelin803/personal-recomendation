import os

#ratings_1有10000054条记录


def count(input_file):
    if not os.path.exists (input_file):
        return
    linenum = 0
    fp = open (input_file)
    for line in fp:
        linenum += 1
        item = line.strip ().split (',')
    return linenum

if __name__=='__main__':
    linenum=count('../data/ratings_1.txt')
    print(linenum)



