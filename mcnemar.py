import sys
import pdb
import rpy2.robjects as ro


def lines_right(line1, line2):
    if line1[1] == line1[2]:
        l1_right = True
    else:
        l1_right = False
    if line2[1] == line2[2]:
        l2_right = True
    else:
        l2_right = False
    return l1_right, l2_right

both_right = 0
both_wrong = 0
only_1_right = 0
only_2_right = 0

for line1, line2 in zip(open(sys.argv[1], 'r'), open(sys.argv[2], 'r')):
    line1 = line1.strip().split('\t')
    line2 = line2.strip().split('\t')
    results = lines_right(line1, line2)
    if results == (False, False):
        both_wrong += 1
    elif results == (True, False):
        only_1_right += 1
    elif results == (False, True):
        only_2_right += 1
    elif results == (True, True):
        both_right += 1
    else:
        print('ERROR\n', line1, '\n', line2)


"""
r stuff:
    x = matrix(c(1,2,3,4), 2, 2)
    mcnemar.test(x)
"""
rstring = "mcnemar.test(matrix(c("
rstring += str(both_right)
rstring += ", "
rstring += str(only_1_right)
rstring += ", "
rstring += str(only_2_right)
rstring += ", "
rstring += str(both_wrong)
rstring += "), 2, 2))"
print(ro.r(rstring))

