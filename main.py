'''some inputs'''
a = [[0, 1], [1, 2], [2, 2], [2, 6], [2, 4], [4, 6], [6, 8], [4, 9], [9, 11], [8, 10]]
b = [[0, 1], [1, 1], [1, 1], [1, 1]]
c = [[0, 0], [2, 2], [2, 4], [4, 5], [5, 7]]


def main(n, array):
    '''

    :param n: # of lift
    :param array: plan of lifts [[a,b],[c,d], ...]
    :param chain: possible chain of lifts [[a,b],[b,c],[c,c] ...]
    :param last : last element of chain [c,c][1] (current floor)
    :return: size of the longest chain
    we always start at ground floor -->> chain=[[0,0]]
    '''

    '''check if we can add element to chain'''

    def check(array, last):
        r = False
        for i in array:
            if i[0] == last:
                r = True
                break
        return r

    '''calculate size of all chains for given array in recursive style'''

    def chain_size(array, chain):
        last = chain[-1][-1]
        if check(array, last):

            for lift in array:
                if lift[0] == last:
                    array_ = list(array)
                    chain_ = list(chain)
                    chain_.append(lift)
                    array_.remove(lift)

                    if chain_size(array_, chain_) != None:
                        '''it is possible that some chain have 0 size and returns None'''
                        results.append(chain_size(array_, chain_))

        else:
            return len(chain) - 1

    '''RUN CODE'''
    results = []
    chain = [[0, 0]]
    if 0 in [lst[0] for lst in array]:
        '''we check if there any lift at ground floor'''
        chain_size(array, chain)  # run principal part
        return max(results)
    else:
        return 0


print(main(7,a))
