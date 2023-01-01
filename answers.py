r"""
Use this module to write your answers to the questions in the notebook.

Note: Inside the answer strings you can use Markdown format and also LaTeX
math (delimited with $$).
"""

# ==============
# Part 1 (Backprop) answers

part1_q1 = r"""
**Your answer:**

1.A.This Jacobian represents the differential of every entry in the input (X) for every entry in the output (Y). Our input (X) size is 64x1024. Our output (Y) size is 64x512 and Y is differentiable at every point (Y is linear).
Thus the jaicobian is a **4D matrix of size 64x1024x64x512**.

1.B.**Yes**, all the elements where Xi and Yj are not from the same batch are zeros, that is, there is 64x1024x512 nonzero elements by definition from 64x1024x64x512 ==> 1/64 elements are nonzero by definition ==> the jaicobian in sparse.

1.C.**No**, we can conclude from the previous section that ($\delta\mat{X}$)=($\delta\mat{Y}$)$\pderiv{\mat{Y}}{\mat{X}}$ actually is equal to ($\delta\mat{X}$)=($\delta\mat{Y}$)$W$

2.A.This Jacobian represents the differential of every entry in the weights (W) for every entry in the output (Y). Our weights (W) size is 512x1024. Our output (Y) size is 64x512 and Y is differentiable at every point (Y is linear).
Thus the jaicobian is a **4D matrix of size 512x1024x64x512**.

2.B.**Yes**, all the elements where i!=j for Yj = XiWlk are zeros, that is, there is 64x512x1024 nonzero elements by definition from 512x1024x64x512 ==> 1/512 elements are nonzero by definition ==> the jaicobian in sparse.

2.C. **No**, we can conclude from the previous section that ($\delta\mat{Wlk}$)=($\delta\mat{Y}$)$\pderiv{\mat{Y}}{\mat{Wlk}}$=$\sum_{}^{} $($\delta\mat{Yj}$)$\pderiv{\mat{Yj}}{\mat{Wlk}}$=($\delta\mat{Yj}$)$Xi$, so simply we can actually calculate ($\delta\mat{W}$)=$(\delta\mat{Y})^TX$

"""

part1_q2 = r"""
**Your answer:**

**No** it's not required, the back-probagation is such an algorithm to train neural networks with decent-based optimization, it is efficient and widly used, but there exist other algorithms that can do its job, i.e. there are alternative algorithms that could be used to train neural networks with decent-based optimization such as equilibrium-propagation algorithm.

"""


# ==============
# Part 2 (Optimization) answers


def part2_overfit_hp():
    wstd, lr, reg = 0, 0, 0
    # TODO: Tweak the hyperparameters until you overfit the small dataset.
    # ====== YOUR CODE: ======
    wstd, lr, reg = 0.1, 0.05, 0.1
    # ========================
    return dict(wstd=wstd, lr=lr, reg=reg)


def part2_optim_hp():
    wstd, lr_vanilla, lr_momentum, lr_rmsprop, reg, = (
        0,
        0,
        0,
        0,
        0,
    )

    # TODO: Tweak the hyperparameters to get the best results you can.
    # You may want to use different learning rates for each optimizer.
    # ====== YOUR CODE: ======
    wstd = 2e-1
    lr_vanilla = 4e-2
    lr_momentum = 3e-3
    lr_rmsprop = 2.7e-4
    reg = 2e-3
    # ========================
    return dict(
        wstd=wstd,
        lr_vanilla=lr_vanilla,
        lr_momentum=lr_momentum,
        lr_rmsprop=lr_rmsprop,
        reg=reg,
    )


def part2_dropout_hp():
    wstd, lr, = (
        0,
        0,
    )
    # TODO: Tweak the hyperparameters to get the model to overfit without
    # dropout.
    # ====== YOUR CODE: ======
    wstd = 1e-3
    lr = 3e-4
    # ========================
    return dict(wstd=wstd, lr=lr)


part2_q1 = r"""
**Your answer:**

1.**Yes** it matches what we expected to see. The model without dropout ovefitted the training data and acheived the highest accuracy on it wherease acheived the worst accuracy on the test data, while the models trained with dropout are more generalized, thus their accuracy on he trainig differ from their accuracy in the test less than the model without dropout differ, this works for the loss too, we can see that the loss of the model without dropout on the training set decreases very nice and smooth and it has the best loss graph on traniing, but it has the worst one on the test, whereas the loss of the models with dropout decreases roughly, it maintaned close loss values on both train and test set, as expected :)

2.From the graphs we can see that despite the two acheivements of the two modeles are close to each other, the graphs of the model with high-dropout setting were changing more roughly than the graphs of the low-droput setting model. This was expected, beacause the model with high-dropout setting may drop more neurons and thus it will be harder for it to converge or even progress smoothly.

"""

part2_q2 = r"""
**Your answer:**

**Yes** it's possible. The cross-entropy loss function is calculated depending on the output scores (its value affected from how high the score for the correct class is and how low the score is for wrong classes), while the accuracy is calculated depending on the number of correct predictions. The output class scores changes in the process of training, and it can increase for wrong classes while the correct score still the leading one, in this case the loss function may increase, and the number of correct classes may increase too and so the accuracy also increases.

"""

part2_q3 = r"""
**Your answer:**

1.Back-propagation is the process of calculating the derivatives of the loss function. The gradient descent is the process of descending through the gradient, i.e. adjusting the parameters of the model to go down through the loss function in order to minimize the loss. Back-probagation and gradient descent complete each other. Gradiant descent needs back-probagation (or another alternative algorithm) to calculate the update the model's parameters.

2.The main difference between Gradient Descent (GD) and Stochastic Gradient Descent (SGD) is that in GD **ALL** the train data is used as one batch to perform the forward and backward passes, while in the SGD each time a **SUBSET** (small batch) of the train data is used for these passes.
In the GD the weights gets adjusted one time in each epoch using all the data set at once, while in the SGD each batch of data adjust the weighte separetly. The adjustment according to GD is computationally very expensive since we need to go on all the data "at once" and to save it in the memory, however it stills a great option for convex or relatively smooth error manifolds and scales well with the number of features. The weight adjusment according to SGD is more sensetive beacause we look at small sample of data and adjust the weights according to it, but it is much faster, and with good choosing of lr value the sensitivity of this option gets negligible.

3.SGD is Faster and less computationally expensive than GD. SGD has smaller escaping time than ADAM-alike adaptive algorithms and tends to converge to flatter minima whose local basins have larger Radon measure, explaining its better generalization performance. 

4.A.**Yes** this approach might produce a gradient equivalent to GD. The final loss will be equal to the original loss as the GD is performed, for example if our loss function is cross entropy, then forwarding |D|/n samples each time yeilds loss =  $\sum_{n=1}^{|D|/n} -y^*log(y)$ and summing them over all the forward passes yeild $\sum_{n=1}^{n}\sum_{n=1}^{|D|/n} -y^*log(y)$ = $\sum_{n=1}^{|D|} -y^*log(y)$. And since the backward pass is applyed on the total loss with respect to all data it will update the model's weight same as if GD is applied.

4.B. There are other factors that need space in the memory, such as intermediate calculations — values from the forward pass that are temporarily stored in GPU memory and then used in the backward pass, the workspace — temporary memory for local variables of kernel implementations.
These might be why he got a memory error since he didn't take these factors into consideration.

"""


# ==============


# ==============
# Part 3 (MLP) answers


def part3_arch_hp():
    n_layers = 0  # number of layers (not including output)
    hidden_dims = 0  # number of output dimensions for each hidden layer
    activation = "none"  # activation function to apply after each hidden layer
    out_activation = "none"  # activation function to apply at the output layer
    # TODO: Tweak the MLP architecture hyperparameters.
    # ====== YOUR CODE: ======
    n_layers=4
    hidden_dims=16
    activation="relu"
    out_activation="none"
    # ========================
    return dict(
        n_layers=n_layers,
        hidden_dims=hidden_dims,
        activation=activation,
        out_activation=out_activation,
    )


def part3_optim_hp():
    import torch.nn
    import torch.nn.functional

    loss_fn = None  # One of the torch.nn losses
    lr, weight_decay, momentum = 0, 0, 0  # Arguments for SGD optimizer
    # TODO:
    #  - Tweak the Optimizer hyperparameters.
    #  - Choose the appropriate loss function for your architecture.
    #    What you returns needs to be a callable, so either an instance of one of the
    #    Loss classes in torch.nn or one of the loss functions from torch.nn.functional.
    # ====== YOUR CODE: ======
    lr, weight_decay, momentum = 0.005, 0.01, 0.9
    loss_fn = torch.nn.CrossEntropyLoss()
    # ========================
    return dict(lr=lr, weight_decay=weight_decay, momentum=momentum, loss_fn=loss_fn)


part3_q1 = r"""
**Your answer:**

1.The optimization error incurres when not finding exact minimizer of in sample loss. According to the graph the in sample loss is relatively low, that is the optimization error is considered as **low**, however, it could be minimized more (even to equal to zero but this will lead to major overfitting).

2.The generalization error is incurred by minimizing training loss instead of test loss. According to the graph the generalization error is **low**: the test loss and train loss are pretty close so that there isn't a generalization error. 

3.The model has **high** approxmation error in some cases. Our model is restircted to two classes and uses a stricted threashold to decide the classification of each sample, so it is safe to assume that the approximation error is high.

"""

part3_q2 = r"""
**Your answer:**

FNR will be higher.
Noise magnitude of validation data is bigger than training data in generating data procces.
The model function predicts the lable of the samples to be positive(1) or not and uses threshold=0.5, FNR is higher on validation because of larger noise making the output of the model less 1.

"""

part3_q3 = r"""
**Your answer:**

1.When the focus is to decrease the further costs testing  when 'positive' prediction in our model, FPR should be smallest as possible. 
It doesn't matter a lot if the model detected one patient with the desease as  "negative" because of the disease can develop obvious non-lethal symptoms and then she will be well treated at a low cost. 
Therefore, the ROC curve left bottom point should be chosen so FNR can be higher.

2.In this case, in order to decrease the loss of life we will minimize the the FNR must, and will allow more classifying patients as positive, therefore the ROC curve left top point should be chosen.

"""


part3_q4 = r"""
**Your answer:**

1. By increasing of the width, the model getting more combination types of input data,
 so its king of generating an  nonlinear decisions boundaries and also have a better prediction 
 performance.

2. In one row, by depth increasig, the model abstracting higher hierarchy feature of input data,
 its could be reagarded as the transfomation of the input. 
, but also an deeper structure can help get nonlinear  boundaries which is helpful for performance 
improving  of this classification problems.

3. The latter one is deeper and has good width relatively design ,
therefore the latter structure in both pairs has better performance, 
which are better for nonlinear characterization. 
Both pairs have total parameters different number.

4. Threshold optimization precedure using to get  on lowest FPR and highest TPR  based on the validation dataset, 
from the time of the test data is regarded as the representative training/validation data,
according that  applying the optimal threshold in test going to improve the accuracy model prediction,
 which is proved by the mlp_experiment.

"""
# ==============
# Part 4 (CNN) answers


def part4_optim_hp():
    import torch.nn
    import torch.nn.functional

    loss_fn = None  # One of the torch.nn losses
    lr, weight_decay, momentum = 0, 0, 0  # Arguments for SGD optimizer
    # TODO:
    #  - Tweak the Optimizer hyperparameters.
    #  - Choose the appropriate loss function for your architecture.
    #    What you returns needs to be a callable, so either an instance of one of the
    #    Loss classes in torch.nn or one of the loss functions from torch.nn.functional.
    # ====== YOUR CODE: ======
    loss_fn = torch.nn.CrossEntropyLoss()
    lr, weight_decay, momentum = 0.003, 0.01, 0.9 
    # ========================
    return dict(lr=lr, weight_decay=weight_decay, momentum=momentum, loss_fn=loss_fn)


part4_q1 = r"""
**Your answer:**

1. 
Number of the bottleneck block: 
$layer1 + layer 2 + layer 3: (1*1*256+1)*64 + (3*3*64+1)*64 + (1*1*64+1)*256 = 70016.$
number of regular block:
 $layer1 + layer 2: (3*3*256+1)*256 + (3*3*256+1)*256 = 1180160.$
the exepectation is there are much less parameters in the bottleneck case (two magnitude orders).

2.
The input size is(256, H, W):
Relu will be considered as 1 basic math operation (floating point operation)

In a regular block:
$layer1 + Relu + layer2 + relu =$ 
$3*3*256*H*W*256 + 256*H*W + 3*3*256*H*W*256 + 256*H*W= 1,180,160*H*W$ 

In a bottleneck block:
$layer1 + Relu + layer2 + Relu + layer3 + skip connection + Relu =$ 
$1*1*256*H*W*64 + 64*H*W + 3*3*64*H*W*64 + 64*H*W + 1*1*64*H*W*256 + 256*H*W + 256*H*W = 70,272*H*W$

3. 
The regular block receptive field is 5X5.
The bottleneck receptive field is 3X3. 
Across the feature maps, thus have the same ability to combine
 the output and both of the models will use all the input channels.

"""

# ==============

# ==============
# Part 5 (CNN Experiments) answers


part5_q1 = r"""
**Your answer:**

The best depth is L=4 with best test accuracy because of the network
has appropriate depth and the learn ability of more complex features.

L=2 - L=16:
The depth increase got improves and then damages the accuracy.
From some depth threshold, the network becomes too deep and starts to have bad influence on the training process.

 L=16 and L=8 :
The model has learnt nothing and the learning process dosent efficient at all. 
probably reasons might be too many pooling layers that vanish the outputand gradient vanishing.
possible solutions to partially fix this include padding the input to increase the dimensions of the output, 
and adding skip connections, like in residual blocks.

"""

part5_q2 = r"""
**Your answer:**

L=8:
the training/learning process is damaged due to too deep network, no matter what K is.

L=2:
Training results with the filter number of (64, 128) are better.

K=128:
The worse Training results performance. 
Seems like an over-fitting caused by too complex features learned with the  channels high number.

Additionally could indicated that is without proper regularization and deep network (large L) with many filters results in too complex network,is over-fit the training set. 
is a relationship between k (the number of filters) and the L (depth).
In out experiment the best result corresponds to L=4 and K=128.

"""

part5_q3 = r"""
**Your answer:**

L=1:
The best performance resaults

L=2, L=3 , L=4:
the networks cannot learn anything due to without a high relationship between fielternumber and depth
 or may too deep architechture.

"""

part5_q4 = r"""
**Your answer:**

(L=8 K=32) and (L=2 K=64-128-256)is the best performance goes, with ilter number and proper depth.

resnet architecture is applied for this experiment, crossing the depth limilation and can get 
high accuracy in very deep architechture is the most obvious feature resnet.

Residual blocks keeps the dimensions of the outputs along the network, what can helps overcome the 
vanishing of the output through deep networks. 

Additionlly, for previous experiments compared ,increases learning ability even when the net is 
really deep like K=[64,128,256] or L=8  and the resnet significantly reduces over-fitting problem.

"""


# ==============

# ==============
# Part 6 (YOLO) answers


part6_q1 = r"""
**Your answer:**

1.The model detected the dolphines images very bad, it didn't detect any dolphine, instead it detected two birds and one person!
The detection of the second image is not good enough too, in that image there is three dogs and one cat, but the model detected only two dogs correctly, and detected the third dog as a cat, and didn't detect the cat!

2.In the dolphine image, the model's failure bassiclly results from that the model used was trained on the COCO dataset, which doesn't incluse a dolphine class, so it is basiclly impossible to classify the dolphins correctly even if their image was clearer bacause the model is supervised and doesn't have such a class option. In addition to that, even if the model was trained on dataset that includes such class, the model still may fail in the detection of this image for reasons related to the image conditions: the right two dolphins appear overlapping which makes detecting them more difficult, illumination - the dolphins losted there colors in the image so detecting them now is based only on there shapes, in addition to that, the image was taken when the dolphins jumped from the see and they completely appear in the sky, this may caused the model to classify two of them as a birds, the third may be classified as a person because of the overlapping which made their bottoms seem as a person's legs. 
These issues might be avoided bassiclly by training the model on larger and more genaral dataset and by providing such images to the training set. And by applying some preprocessing to the image to solve the illumination problem and to give the dolphoines some colors.

For the cat&dogs image, we can see that the left dog and the cat were classified as one object, this may reason from that they are located very close to each other, so the bounding boxes from each cell of them may be overlapped. This issue might be avoided by using smaller bouncing boxes.

"""


part6_q2 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""


part6_q3 = r"""
**Your answer:**

The model detection is not good enough, it had some pitfall in detectiong the objects in the images:
1. In the first image (i.e. the 5 newborn cats) the model detected only one cat and a dog! If we look at this picture, we can see that the cats are patially occluded, this may be the main reason for this failure. 
2. The second image (i.e. the two cats setting on thei chair) has illumination issues, this lead the model to detect only the cats' chair, without detecting the two cats setting on it!
3. The third image (i.e. the two ice-cream bowls on patterned table) has textured background, and if we look carfully to the model's result on this image we can see that it didn't detect the top spoon, this maight resulted from the textured background.
"""

part6_bonus = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""