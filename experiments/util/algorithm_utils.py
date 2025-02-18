import numpy as np

# import heapq
# from operator import itemgetter


# def arg_top_k(a_iterable, top_k):
#     a_dict = {}
#     for index, a in enumerate(a_iterable):
#         a_dict[index] = a
#     nlargest_list = heapq.nlargest(top_k, a_dict.items(), key=itemgetter(1))
#     return [ e[0] for e in nlargest_list ]

def arg_top_k(a_iterable, top_k):
    a_tuple_list = [(a, np.random.uniform(0.0, 1.0), index) for index, a in enumerate(a_iterable)]
    return [ index for a, b, index in sorted(a_tuple_list, reverse=True)[:top_k] ]
