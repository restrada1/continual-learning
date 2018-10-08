import visdom
import torch
import numpy as np
vis = visdom.Visdom()
vis.close()
trace = dict(x=[1, 2, 3], y=[4, 5, 9], mode="markers+lines", type='custom',
             marker={'color': 'red', 'symbol': 104, 'size': "10"},
             text=["one", "two", "three"], name='1st Trace')
layout = dict(title="First Plot", xaxis={'title': 'x1'}, yaxis={'title': 'x2'})

vis._send({'data': [trace], 'layout': layout, 'win': 'mywin'})

# plt=vis.line(Y=torch.randn(5))


vis.bar(
    X=np.abs(np.random.rand(5, 1)),
    # opts=dict(
    #     stacked=False,
    #     #legend=['Facebook'],
    #     rownames=['2012', '2013', '2014', '2015', '2016']
    # )
)


wiz=vis.bar(
    X=np.random.rand(10),
    # opts=dict(
    #     stacked=False,
    #     legend=['The Netherlands', 'France', 'United States']
    # )
)

# l=[1]
# for i in range(10,20):
#     l.append(i)
#     print(l)
#     vis.bar(X=np.array(l),win=wiz)

# # vis.line(Y=np.random.rand(10))

# # line updates
# win = vis.line(
#     X=np.column_stack((np.arange(0, 10), np.arange(0, 10))),
#     Y=np.column_stack((np.linspace(5, 10, 10), np.linspace(5, 10, 10) + 5)),
# )

# vis.line(
#    X=np.column_stack((np.arange(0, 10), np.arange(0, 10))),
#     Y=np.column_stack((np.linspace(5.5, 10, 10), np.linspace(5.5, 10, 10) + 5)),
#     win=win,
#     update='append'
# )

# WINDOW_CASH={}
# def visualize_scalars(scalars, names, title, iteration, env='main'):
#     assert len(scalars) == len(names)
#     # Convert scalar tensors to numpy arrays.
#     scalars, names = list(scalars), list(names)
#     scalars = [s.cpu() if isinstance(s, CUDATensor) else s for s in scalars]
#     scalars = [s.numpy() if hasattr(s, 'numpy') else np.array([s]) for s in
#                scalars]
#     multi = len(scalars) > 1
#     num = len(scalars)

#     options = dict(
#         fillarea=True,
#         legend=names,
#         width=400,
#         height=400,
#         xlabel='Iterations',
#         ylabel=title,
#         title=title,
#         marginleft=30,
#         marginright=30,
#         marginbottom=80,
#         margintop=30,
#     )

#     X = (
#         np.column_stack(np.array([iteration] * num)) if multi else
#         np.array([iteration] * num)
#     )
#     Y = np.column_stack(scalars) if multi else scalars[0]

#     vis(env).updateTrace(X=X, Y=Y, win=WINDOW_CASH[title], opts=options)
   
#test single line

# win = vis.line(
# X=np.array([1,2,3,4,5]),
# Y=np.array([1,2,3,4,5]),
# win="test",
# name='Line1',
# )

# for i in range(100000):
#     vis.line(
#         X=np.array([i + 1]),
#         Y=np.array([i + 1]),
#         win="test",
#         name='Line1',
#         update='append',
#     )