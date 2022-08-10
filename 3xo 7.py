'''inputs'''
str1='hello'
bondaries=[[1,5], [1,2],[2,5]]


def main(str1,bondaries):
    '''

    :param str1: plan of shop
    :param bondaries: bondaries to look for
    :param results: # of steps for each bondaries
    :return: max of results
    '''
    results=[]
    for bondary in bondaries:
        str=str1[bondary[0]-1:bondary[1]]
        str_sorted=sorted(str)
        steps=0

        for letter in str_sorted:
            while(letter!=str[(steps%bondary[1])]):
                    steps+=1

        results.append(steps)
    return results

print(main(str1,bondaries))