# ID2223 Scalable ML/DL: Plan for an A (detailed study notes)

## How the grade works

- Labs (CNN + LLM), essay + presentation, and the final project are all **Pass/Fail**. You must pass them, but they don't change your letter.
- **The letter comes only from the exam:** 20 multiple-choice questions, 1 point each, no aids allowed. **An A needs 18 or more**, so you can miss at most 2.
- Tasks 1–3 are meant for groups of three. You're doing them solo, so **email the examiner (Amir Payberah)** to get that approved and ask whether the scope is reduced.
- Nearly every exam question asks "why does X exist / what problem does X solve". So these notes explain the *why* behind each idea, not just the definition. If you can explain the problem a technique solves, you can eliminate the wrong options even on a question you've never seen.

**How to use these notes:** read one section, close the doc, and explain it out loud as if teaching a friend. Then take the quiz.

---

# LECTURE 2: Regression and Classification

## 2.1 What linear regression is trying to do
You have m houses. For each one you know some **features** (living area x₁, number of bedrooms x₂) and the **target** (price y). You want a function that predicts the price of a house you haven't seen.

Linear regression assumes the price is roughly a **weighted sum** of the features:

ŷ = w₁x₁ + w₂x₂ + b

- **w₁** answers: *how much does the price go up for each extra m²?* If w₁ = 0.3 (in thousands), each m² adds 300.
- **w₂** answers: *how much does each extra bedroom add?*
- **b (intercept/bias)** is the baseline. Without it, the prediction for a house with 0 area and 0 bedrooms would have to be exactly 0, which forces the line through the origin. Real data rarely behaves like that, so b lets the model shift up or down.

"Learning" means finding the w and b that make ŷ close to y on the training data.

**Example.** Suppose the model is price = 0.3·area + 10·bedrooms + 50. For 100 m² with 3 bedrooms, it predicts 30 + 30 + 50 = **110**.

## 2.2 The bias trick (x₀ = 1)
**The problem:** b behaves differently from the weights, because it doesn't multiply any feature. So you'd need separate formulas for b everywhere: in the prediction, in the gradient, and in the update.

**The trick:** any constant can be written as b·1. So give every example an extra "feature" x₀ that always equals 1, and rename b to w₀:

ŷ = w₀·1 + w₁x₁ + w₂x₂ = **wᵀx**, with **x** = [1, x₁, x₂] and **w** = [w₀, w₁, w₂]

**Check with numbers:** **x** = [1, 100, 3] and **w** = [50, 0.3, 10] give 50 + 30 + 30 = 110. That's the same answer, because the model is unchanged. Only the notation is simpler.

**What you gain:** one formula for everything. The prediction is wᵀx, the gradient is ∇J(w), and the update is w ← w − η∇J(w). The bias is learned automatically as just another weight. For the whole dataset, put a column of 1s in front of the data matrix X, and every prediction at once is **ŷ = Xw**, which is how NumPy and GPUs compute it.

The **bias neuron** in neural networks (Lecture 3), which "always outputs 1", is the same trick.

**Exam (re-exam Q1):** "Why x₀ = 1?" → *to include the bias term in the weight vector.* It doesn't normalize anything (a constant 1 rescales nothing). It doesn't remove the need for gradient descent. And it doesn't reduce overfitting, because the model and its number of parameters are exactly the same.

## 2.3 Measuring "how wrong": residuals and MSE
The **residual** for example i is y⁽ⁱ⁾ − ŷ⁽ⁱ⁾, how far off one prediction is. You need one number for the whole dataset, so you can compare different choices of w.

**Why not just add up the residuals?** Positive and negative errors cancel. Errors of +50 and −50 sum to 0, which looks perfect when it isn't.

**Mean Squared Error:** J(w) = (1/m) Σ (ŷ⁽ⁱ⁾ − y⁽ⁱ⁾)²

Why squaring is a good choice:

1. **It's always positive**, so nothing cancels.
2. **Large errors count much more.** An error of 10 contributes 100, and an error of 1 contributes 1. The model is pushed hard to avoid big mistakes.
3. **It's differentiable everywhere** (a smooth parabola), so gradient descent can follow its slope. |error| has a sharp corner at 0, which is less convenient.
4. **For linear regression it's convex**, a single bowl with one bottom, so gradient descent can't get stuck in a wrong valley.

**Example:** three houses with errors 2, −1 and 5 give MSE = (4 + 1 + 25)/3 = 10. Notice how the single error of 5 dominates.

**Exam (Jan Q3):** "Why is MSE reasonable for linear regression?" → *it penalizes large errors heavily and is differentiable.* The wrong options: it guarantees nothing about fairness (see 2.6). It doesn't give sparse solutions (that's L1, in Lecture 4). "Prevents local minima" isn't a property of MSE in general. Convexity here comes from MSE *combined with a linear model*, and it fails with a sigmoid (see 2.8).

## 2.4 Gradient descent: how we actually find w
**Intuition (from the slides):** you're lost on a mountain in thick fog, and you can only feel the slope under your feet. To reach the valley floor, you step in the steepest downhill direction and repeat.

Here, the "mountain" is the cost surface J(w): each point is one choice of weights, and the height is how wrong that choice is.

**The algorithm:**

1. Start with random w.
2. Compute the **gradient** ∇J(w), a vector of slopes, one per weight. It points **uphill**, toward increasing error.
3. Step the opposite way: **w ← w − η ∇J(w)**
4. Repeat until the gradient is about 0 (flat ground, the minimum) or you run out of steps.

**Why the gradient tells you the direction:** ∂J/∂wⱼ says how much the error changes if you increase wⱼ a little. If it's positive, increasing wⱼ makes things worse, so you decrease it. The minus sign does exactly that.

**A tiny worked example.** One weight, J(w) = (w − 3)², so the minimum is at w = 3. The gradient is 2(w − 3).

- Start at w = 0 with η = 0.1. The gradient is −6, so w ← 0 − 0.1·(−6) = 0.6. Next, the gradient is −4.8, so w = 1.08, and so on. It creeps toward 3. ✔
- With η = 1.1, the gradient at w = 0 is −6, so w ← 0 + 6.6 = 6.6. Now you're further away than when you started (distance 3.6 instead of 3). The next step overshoots even more. **Divergence.** ✘

## 2.5 Learning rate and the three variants of gradient descent
**Learning rate η** is the step size.

- **Too small:** each step barely moves, so you need thousands of iterations.
- **Too large:** you jump over the valley onto the opposite slope, possibly higher up. The error grows instead of shrinking, and the algorithm **diverges** (as in the η = 1.1 example).

**Exam (re-exam Q2):** "What happens if the learning rate is too large?" → *the algorithm may diverge.* Not "faster and more stable" (it's unstable). Not "underfitting" (that's about model capacity). Not "the gradient becomes zero" (it usually grows).

**How much data do you use to compute each gradient?**

| Variant | Each step uses | Why you'd use it | Downside |
|---|---|---|---|

| **Batch GD** | all m examples | exact gradient, smooth path to the minimum | one step needs a full pass over the data, which is far too slow for millions of examples |
| **Stochastic GD (SGD)** | 1 random example | very cheap steps, starts improving immediately; the noise can help it escape shallow local minima | noisy and zig-zagging; it keeps bouncing around the minimum instead of settling |
| **Mini-batch GD** | e.g. 32–256 examples | the middle ground: steadier than SGD, much cheaper than batch, and matrix operations on a batch run efficiently on GPUs | — (this is the deep-learning default) |

**Feature scaling:** if area ranges from 0 to 300 and bedrooms from 1 to 5, the cost surface becomes a long, narrow valley. Gradient descent zig-zags across the steep direction and crawls along the flat one. Scaling all features to similar ranges (for example, standardizing to mean 0 and std 1) makes the bowl round, so the steps head straight to the minimum.

## 2.6 Critical view: MSE and fairness
**The example from the slides:** a model predicts student performance to decide where resources go. It has a very low MSE, but it was trained mostly on data from well-resourced schools. It **systematically underpredicts** students from under-resourced schools, who then get fewer resources, which reinforces the original inequality.

**Why this happens:** MSE averages the error over *everyone*. If one group is 90% of the data, the model can do great on them and badly on the other 10%, and still have a low average. **MSE minimizes total error, not fairness.**

**What to ask:** *"Better for whom?"* Check the error separately for each group, and consider group-weighted metrics that balance fairness and accuracy.

**Exam:** any option claiming MSE or a low loss "guarantees fairness across subgroups" is wrong.

## 2.7 Classification: why linear regression fails
**The task:** predict cancer (1) or no cancer (0) from tumor size. We want a **probability** of class 1.

**Why not fit a straight line?** (1) A line's outputs aren't limited to [0,1], so it can predict 1.7 or −0.4, which aren't probabilities. (2) One very large tumor in the data tilts the whole line and moves the decision boundary, misclassifying other examples. The slides call this a "bad model".

## 2.8 Logistic regression
**Fix:** keep the weighted sum t = wᵀx, but pass it through a function that squashes any number into (0,1):

**Sigmoid:** σ(t) = 1 / (1 + e⁻ᵗ)

- t very negative → e⁻ᵗ is huge → σ ≈ 0
- t = 0 → σ = 1/2
- t very positive → e⁻ᵗ ≈ 0 → σ ≈ 1

So p̂ = σ(wᵀx) can be read as "the probability this is class 1". **Decision rule:** predict 1 if p̂ ≥ 0.5, which happens exactly when wᵀx ≥ 0. So the decision boundary is still a straight line (or plane) where wᵀx = 0. The sigmoid only turns the distance from that line into a confidence.

**Example:** with wᵀx = 2, p̂ = 1/(1 + e⁻²) ≈ 0.88, so predict cancer with 88% confidence.

**Exam (re-exam Q3):** "Correct statement about the sigmoid?" → *it maps real numbers to probabilities.* Its range is (0,1), not [−1,1] (that's tanh). It flattens for large inputs; it isn't linear there. It outputs 0.5 only at t = 0.

## 2.9 Why logistic regression uses log loss instead of MSE
**Option 1, MSE:** J = (1/m)Σ(σ(wᵀx) − y)². Combined with the sigmoid, this surface is **non-convex**: it has bumps and several valleys. The slides' test for non-convex is that the line joining two points on the curve crosses the curve. Gradient descent can then settle in a **local minimum** that isn't the best solution.

**Option 2, log loss (cross-entropy):**

- If y = 1: cost = −log(p̂)
- If y = 0: cost = −log(1 − p̂)
- Combined: J(w) = −(1/m) Σ [ y log p̂ + (1 − y) log(1 − p̂) ]

**Why it behaves well:** look at y = 1.

- p̂ = 0.99 → cost = −log 0.99 ≈ 0.01, a confident and correct prediction that's barely penalized.
- p̂ = 0.5 → cost ≈ 0.69.
- p̂ = 0.01 → cost = −log 0.01 ≈ 4.6, a confident wrong answer that's punished hard.
As p̂ → 0, the cost → ∞. So the model is strongly pushed away from confident mistakes, and with the sigmoid this cost is **convex** (one global minimum).

## 2.10 Multi-class: softmax regression
With k classes (say digits 0–9), compute one score per class, sⱼ = wⱼᵀx, then turn the scores into probabilities with **softmax**:

p̂ⱼ = e^{sⱼ} / Σₗ e^{sₗ}

- The exponential makes every value positive.
- Dividing by the sum makes the values **add up to 1**, giving a probability distribution over the classes.
- Bigger scores get disproportionately more probability.

**Example:** scores [2, 1, 0] → e² ≈ 7.39, e¹ ≈ 2.72, e⁰ = 1, with a sum of ≈ 11.1 → probabilities ≈ [0.67, 0.24, 0.09]. Predict class 1 (the first one).

The loss is **cross-entropy**, −log(p̂ of the true class), the same idea as log loss.

**Exam (Jan Q4):** the best pairing for multi-class is **softmax + cross-entropy**. Sigmoid treats each output independently, so the outputs don't sum to 1 and it doesn't give a distribution over mutually exclusive classes. MSE with probability outputs gives weak gradients and the non-convexity problem described above.

---

# LECTURE 3: Deep Feedforward Networks

## 3.1 From biological neurons to the perceptron

- **Biology:** a neuron fires when it receives enough signals within a few milliseconds. Billions of neurons, each connected to thousands of others, form the network. ANNs borrow this idea loosely.
- **McCulloch–Pitts neuron (1943):** binary inputs and a binary output. It fires if enough inputs are on, so it can compute AND, OR and NOT.
- **Linear Threshold Unit (LTU):** inputs are real numbers, each with a weight. It computes z = wᵀx and outputs **step(z)**, which is 1 if z ≥ 0, otherwise 0. This is exactly logistic regression with the sigmoid replaced by a hard step.
- **Perceptron:** a single layer of LTUs. Input neurons simply pass the inputs through, plus a **bias neuron** that always outputs 1 (the bias trick from 2.2).

## 3.2 How a perceptron learns
**Hebb's rule:** "neurons that fire together wire together". If neuron A often helps trigger neuron B, strengthen the connection between them.

**Perceptron rule:** for each training example, predict, then update:

w_{i,j} ← w_{i,j} + η (y_j − ŷ_j) x_i

Read it case by case:

- **Correct prediction** (y = ŷ): the bracket is 0, so nothing changes.
- **Predicted 0 but the answer was 1:** the bracket is +1, so the weights on active inputs (x_i > 0) increase, and next time z is larger and more likely to fire.
- **Predicted 1 but the answer was 0:** the bracket is −1, so those weights decrease.

The perceptron convergence theorem says that if the classes are linearly separable, this rule finds a separating line.

## 3.3 The XOR problem: why one layer isn't enough
XOR truth table: (0,0)→0, (0,1)→1, (1,0)→1, (1,1)→0.

Plot those four points on a square. The 1s sit on one diagonal and the 0s on the other. **No single straight line has both 1s on one side and both 0s on the other.** XOR is **not linearly separable**.

A perceptron's decision boundary is always one straight line (wᵀx = 0), so it **cannot** learn XOR, however long you train it. The slides show that fitting a linear model to XOR with MSE gives w₁ = w₂ = 0 and b = 0.5, which predicts **0.5 for everything**. The best a line can do is give up.

**The fix: add a hidden layer.** For example:

- Hidden neuron h₁ = OR(x₁, x₂)
- Hidden neuron h₂ = AND(x₁, x₂)
- Output = h₁ AND NOT h₂

In the (h₁, h₂) space, the four points become (0,0)→0, (1,0)→1, (1,0)→1, (1,1)→0, and now one line separates them. The hidden layer **re-represented the input** with non-linear features, so the problem became linearly separable.

**Exam (Jan Q5, re-exam Q4):** the reason a hidden layer solves XOR / the key advantage of an MLP over a perceptron → *it learns non-linear feature transformations / can model non-linear decision boundaries.* The wrong options: depth doesn't reduce overfitting (it tends to increase it). Hidden layers don't replace bias neurons. They don't guarantee faster convergence or lower cost; MLPs are slower and costlier than a perceptron.

## 3.4 The Multi-Layer Perceptron (MLP)

- **Structure:** an input layer, then one or more **hidden layers**, then an output layer. Every layer except the output has a bias neuron and is **fully connected** to the next. Signals only move forward (no loops), hence the name **feedforward network**. Many hidden layers make it a **deep** network.
- Each neuron does what logistic regression does: a weighted sum, then a non-linear activation. An MLP is many of those stacked together, where later layers build on the features earlier layers found.

## 3.5 Why the activation function matters (two exam traps)
**Trap 1: the step function can't be used with backprop.** Training needs ∂(error)/∂(weight). The step function is flat everywhere, so its derivative is **0**, and at z = 0 it jumps, so the derivative is undefined. Every gradient that passes through a step neuron becomes 0, so no weight before it ever learns. That's why MLPs use smooth functions:

- **Sigmoid**: (0,1), smooth. Its derivative is σ(1−σ), at most 0.25.
- **tanh**: (−1,1), zero-centered. It often trains faster than sigmoid.
- **ReLU** = max(0,z): cheap, with gradient 1 for z > 0.

**Exam (re-exam Q5):** "Why not a step function in hidden layers with backprop?" → *it's not differentiable* (gradient zero or undefined). Being slow, producing non-binary outputs or causing overfitting are all false.

**Trap 2: linear activations make depth useless.** If every layer computes a linear function, then layer 2 applied to layer 1 is W₂(W₁x) = (W₂W₁)x = W′x, which is just **one matrix**. A 100-layer linear network is mathematically the same as a single-layer linear model, and it still can't solve XOR. **Non-linearity is what makes depth add power.**

**Exam (re-exam Q6):** "All activations linear → ?" → *it behaves like a single linear model.* It isn't more expressive. It can still be trained (it's just weak). It doesn't become non-convex (non-linearity is what causes non-convexity).

## 3.6 Choosing the output layer and the loss
The output layer depends on what you're predicting:

| Task | Output layer | Why | Loss |
|---|---|---|---|

| Regression (a price) | linear, no activation | the output can be any real number | MSE |
| Binary (spam or not) | 1 sigmoid neuron | the output is P(class 1) in (0,1) | binary cross-entropy |
| Multi-class (digit 0–9) | k neurons + softmax | the outputs form a distribution that sums to 1 | cross-entropy |

## 3.7 Backpropagation: how an MLP learns
**The problem:** a deep network might have millions of weights. For gradient descent you need ∂E/∂w for **every** weight. Nudging each weight one at a time to see what happens would take millions of forward passes per step, which is impossible.

**The tool: the chain rule.** If y = g(x) and z = f(y), then **dz/dx = (dz/dy)·(dy/dx)**. Effects multiply along a chain. If x affects z through two paths, add the two chains together.

A weight deep in the network affects the error through a chain: weight → that neuron's input → its output → the next layer → … → output → error. So its gradient is the **product of the local derivatives along that chain**.

**The algorithm (for each mini-batch):**

1. **Forward pass:** compute and *store* every neuron's output, layer by layer, and the final error.
2. **Backward pass:** start at the output, where you know how the error depends on each output. Move back one layer at a time. Each layer's gradients are computed from the gradients of the layer after it, plus its local derivatives. Intermediate results are **reused** instead of recomputed, so all gradients cost about as much as one extra forward pass.
3. **Update:** a gradient-descent step on every weight, w ← w − η ∂E/∂w.

**The worked example from the slides.** 2 inputs, 2 hidden neurons (h₁, h₂), 2 outputs (o₁, o₂), sigmoid everywhere, squared error E = Σ ½(target − out)². We want ∂E/∂w₅, where w₅ connects h₁ to o₁.

How w₅ affects E: w₅ → net_o1 (o₁'s weighted input) → out_o1 (after the sigmoid) → E. So:

∂E/∂w₅ = **∂E/∂out_o1** × **∂out_o1/∂net_o1** × **∂net_o1/∂w₅**

1. **∂E/∂out_o1 = (out_o1 − target_o1)**: how wrong the output is. If o₁ is too high, this is positive.
2. **∂out_o1/∂net_o1 = out_o1(1 − out_o1)**: the sigmoid's derivative. It shows how sensitive the output is to its input. Near 0 or 1 this is tiny, which is the root of vanishing gradients (4.3).
3. **∂net_o1/∂w₅ = out_h1**: since net_o1 = w₅·out_h1 + w₆·out_h2 + b, the effect of w₅ is the size of the signal it carries.

Multiply the three factors, then update w₅ ← w₅ − 0.5 × (that product), with η = 0.5.

**In words:** a weight changes in proportion to *how wrong the output is* × *how responsive the neuron is* × *how active its input was*. Make sure you can say that sentence.

For a hidden-layer weight (say w₁, from input x₁ to h₁), the chain is longer. h₁ affects **both** outputs, so you add the contributions through o₁ and through o₂. That's the "two paths" chain rule.

---

# LECTURE 4: Training Deep Feedforward Networks

Deep networks are powerful but hard to train. The slides name **three challenges**, each with its own set of fixes. Learn them as problem → cause → fixes.

## 4.1 Challenge 1: Overfitting
**The problem:** a network with millions of parameters has a huge **degree of freedom**. It can fit almost any dataset, including the noise. It can memorize the training set, getting training error near 0 but high error on new data.

**The general fix:** reduce the effective freedom of the model. There are four techniques.

### Early stopping
Plot the error over training. **Training error** keeps falling, because the model keeps fitting the training data more closely. **Validation error** falls at first (the model is learning real patterns), then reaches a minimum, then **rises** (it's now learning noise specific to the training set). Stop at that validation minimum and keep those weights.

**Exam (re-exam Q7):** the signal to stop → *validation loss starts increasing.* "Training loss stops decreasing" is wrong, because training loss usually keeps decreasing while you overfit. Accuracy rarely reaches 100%, and gradients vanishing is a separate issue.

### L1 and L2 regularization
Add a penalty for large weights: **J_total = J(w) + λ·R(w)**.

**Why large weights mean overfitting:** to pass exactly through every noisy training point, a model needs extreme weights that make the function swing sharply. Penalizing large weights forces a smoother, simpler function. λ sets how strong the penalty is.

- **L2 (ridge):** R = Σ wⱼ². Its gradient is 2wⱼ, so each weight is pulled toward 0 in proportion to its size. Big weights shrink a lot, small ones a little, and they rarely reach exactly 0. The result is that all weights are small.
- **L1 (lasso):** R = Σ |wⱼ|. Its gradient is ±1, a constant pull toward 0 regardless of size, so small weights get pushed **all the way to 0**. The result is **sparse** weights, which acts as automatic feature selection.

### Dropout
At each training step, each neuron is **temporarily removed with probability p** (e.g. 0.5). Its output is set to 0 for that step, and different neurons are dropped on every step.

**The analogy from the slides:** would a company work better if each employee tossed a coin every morning to decide whether to come in? Surprisingly, yes in one sense. No task can depend on one specific person, so knowledge gets spread across the team and nobody is a single point of failure.

**Why it reduces overfitting:**

1. A neuron can't rely on particular other neurons always being present, so it must learn features that are **useful on their own and redundant across neurons**. Memorizing through fragile co-adaptations stops working.
2. With N neurons, there are **2ᴺ** possible thinned networks. Each step trains a different one, all sharing weights. The final network acts like an **average of a huge ensemble**, and ensembles generalize better.

**At inference, dropout is turned off.** All neurons are used, and outputs are scaled so the expected activation matches what the network saw during training.

**Exam (Jan Q6):** why dropout reduces overfitting → *it forces neurons to learn redundant, more robust representations.* It doesn't increase the dataset (that's data augmentation). It doesn't remove layers at inference (all neurons are used). It can't make optimization convex.

### Data augmentation
More data is the best cure for overfitting, but real data is expensive. So **create realistic variations** of the existing examples, such as images that are shifted, rotated, flipped or have brightness changed. A cat rotated by 10° is still a cat. The model learns that these changes don't matter instead of memorizing exact pixels.

## 4.2 Challenge 2: Vanishing and exploding gradients
**The problem:** backprop sends the error from the output back toward the input. In deep networks, the gradients often **shrink as they travel down** the layers. The lower layers get almost no update, so their weights stay near their random initial values and never learn useful features, even though the top layers train fine. That's **vanishing gradients**. The opposite case, where gradients grow layer by layer until the updates are huge and training blows up, is **exploding gradients** (common in RNNs).

**Why it happens (the slides' derivation).** Take a network with just **one neuron per layer**, with activations aⱼ = σ(zⱼ). By the chain rule, the gradient for the first layer is a product with one factor per layer:

∂C/∂b₁ = σ′(z₁) · w₂ · σ′(z₂) · w₃ · σ′(z₃) · w₄ · σ′(z₄) · ∂C/∂a₄

The sigmoid's derivative is **at most 0.25**, and with typical small weights each wⱼ·σ′(zⱼ) is **below 1**. Multiply many numbers below 1 and the product shrinks exponentially. For example, 0.25¹⁰ ≈ 0.000001. That's vanishing. If each factor is above 1 (large weights), the product grows exponentially instead. That's exploding.

There are **four fixes**, each addressing a different part of that product.

### Fix 1: Weight initialization
**Why initialization matters at all:** the cost function of a neural net is non-convex, and where gradient descent starts strongly affects where it ends up. Two requirements:

1. **Break symmetry.** Suppose every weight starts at the same value. Then every neuron in a layer computes the same output, gets the **same gradient**, receives the same update, and stays identical forever. A layer of 100 neurons acts like 1. Random initialization makes the neurons different from the start, so they can learn different features.
2. **Keep the signal flowing in both directions.** If the weights are too small, activations shrink layer by layer on the way forward and gradients shrink on the way back (vanishing). If they're too large, both explode. You want the **variance of each layer's outputs to roughly equal the variance of its inputs**, and the same for gradients going backward.

That's done by scaling the random weights based on how many inputs and outputs each layer has:

- **Glorot (Xavier) initialization:** for sigmoid, tanh or no activation.
- **He initialization:** for ReLU and its variants. ReLU zeros out about half of its inputs, so it needs roughly twice the variance to keep the signal strength.

**Exam (re-exam Q8):** "Why is proper initialization critical?" → *it avoids symmetry and ensures signal flow.* It doesn't affect inference speed or dataset size, and it doesn't eliminate overfitting.

### Fix 2: Non-saturating activation functions
The sigmoid **saturates**: for large |z| it's flat, so σ′ ≈ 0. That's the tiny factor in the product above.

**ReLU(z) = max(0, z)** has derivative exactly **1** for z > 0, so it doesn't shrink the gradient at all on that side. It's also very cheap to compute.

**ReLU's problem, dying ReLUs:** if a neuron's weighted input becomes negative for every example (for instance, after a big update), its output is always 0 and its gradient is always 0. It never updates again, so it's permanently dead. Sometimes half a network ends up dead.

**Fixes for dying ReLUs:**

- **Leaky ReLU** = max(αz, z) with a small α like 0.01. Negative inputs get a small slope instead of 0, so the gradient is never exactly 0 and dead neurons can recover.
- **RReLU:** α is random during training, which also acts as a regularizer.
- **PReLU:** α is **learned** by backprop.
- **ELU:** smooth, with negative outputs that curve toward −α. The average output is closer to 0 and it often converges faster, but it's slower to compute.

### Fix 3: Batch Normalization
**The problem:** during training, every layer's weights keep changing, so the **distribution of inputs to the next layer keeps shifting**. Each layer is chasing a moving target, and if those inputs drift into the sigmoid's flat region, gradients vanish.

**BN's fix:** inside the network, before or after each layer's activation, normalize the values to a consistent range. For each mini-batch B, with m_B examples:

1. Compute the batch mean μ_B and standard deviation σ_B of each feature.
2. **Normalize:** x̂ = (x − μ_B) / √(σ_B² + ε). Now the values have mean ≈ 0 and variance ≈ 1. The tiny ε avoids division by zero.
3. **Scale and shift:** z = γ ⊗ x̂ + β, where γ and β are **learned** parameters (⊗ is element-wise multiplication).

**Why step 3?** Forcing mean 0 and variance 1 might be wrong for that layer. For example, it would keep a sigmoid always in its middle linear region. With learnable γ and β, the network can choose the best scale and offset, and it can even undo the normalization completely if that helps. BN guarantees a *stable* distribution, not necessarily a standard one.

**Benefits:** much less vanishing, larger learning rates are possible, it's less sensitive to initialization, and it has a slight regularizing effect.

**Exam (re-exam Q9):** what BN primarily does → *keeps the inputs to each layer in a stable distribution.* It doesn't remove layers, reduce parameters (it adds γ and β), or "remove noise".

### Fix 4: Gradient clipping
This one is for **exploding** gradients: just cap them. If any gradient exceeds a threshold, clip it during backprop before the update. There are two ways, with different effects:

- **clipvalue = 1.0:** clip *each component* to [−1, 1]. The gradient [0.9, 100] becomes [0.9, 1.0]. The **direction changes**: originally almost entirely along the second axis, now roughly diagonal.
- **clipnorm = 1.0:** if the vector's total length exceeds 1, *rescale the whole vector* to length 1. [0.9, 100] becomes [0.009, 0.9999]. The **direction is preserved** and only the size shrinks.

clipnorm is usually the safer choice, because the step still points in the true downhill direction.

## 4.3 Challenge 3: Training speed (better optimizers)
Plain gradient descent, **w ← w − η∇J**, only looks at the slope *right now*. That causes two problems:

1. On a gentle slope, the gradient is small, so it crawls.
2. In a narrow valley, the gradient points mostly across the valley, so it zig-zags from wall to wall and makes slow progress along the valley floor.

### Momentum
**Idea:** think of a ball rolling downhill. It doesn't just follow the local slope. It **builds up speed** and keeps moving in the direction it's been going.

- m ← β·m − η∇J   (add the current gradient into a running "velocity" vector m)
- w ← w + m       (move by the velocity)
- β is between 0 and 1 (typically 0.9). It's like friction: it controls how much past velocity carries over.

**Why it's faster:** on a consistent slope, the gradients point the same way every step, so m keeps growing. With β = 0.9, the effective step approaches 10× the plain step. In a zig-zag valley, the across-valley components flip sign each step and **cancel out**, while the along-valley components add up. It also has enough "speed" to roll over small bumps and out of shallow local minima.

**Exam (Jan Q7):** the problem momentum primarily addresses → *slow convergence.* Not vanishing gradients (that's initialization, activations and BN). Not overfitting (that's regularization). Not feature scaling.

### Nesterov momentum
This is a small but useful improvement. Compute the gradient not at the current position but **slightly ahead, in the direction momentum is already carrying you**. It's like looking where you're about to land before deciding how to steer. It corrects overshooting earlier and usually converges a bit faster.

### RMSProp
**A different problem:** one global learning rate is a bad fit when parameters see very different gradient sizes. A parameter with consistently large gradients needs small steps or it overshoots. One with tiny gradients needs big steps or it never moves.

**Idea:** give **each parameter its own adaptive learning rate**.

- s ← β·s + (1 − β)·(∇J)²   (a running average of each parameter's *recent* squared gradients)
- w ← w − η · ∇J / √(s + ε)

Dividing by √s means that if a parameter's gradients have been large, its step shrinks, and if they've been small, its step grows. "Recent" matters: because it's an exponentially decaying average, old gradients are forgotten. (An older method, AdaGrad, summed *all* past gradients, so its steps shrank toward zero and training stalled.)

### Adam = Momentum + RMSProp
Adam keeps **both** running averages:

- like momentum, an average of the **gradients**, which gives the *direction* to move;
- like RMSProp, an average of the **squared gradients**, which sets *how big a step* each parameter takes.

The update moves in the momentum direction, scaled per parameter by the RMSProp factor. It works well with little tuning, so it's the default choice in practice.

## 4.4 Summary: problem → fixes

| Problem | Root cause | Fixes |
|---|---|---|

| Overfitting | too much freedom, so the model memorizes noise | early stopping, L1/L2, dropout, data augmentation |
| Vanishing/exploding gradients | a product of many factors < 1 or > 1 in backprop | Glorot/He initialization, ReLU family, batch norm, gradient clipping (exploding) |
| Slow training | plain GD only sees the current slope and uses one global learning rate | momentum, Nesterov, RMSProp, Adam |

**Exam strategy:** for any "what does X fix" question, find X's row in this table. Options naming problems from a *different* row are the distractors.

---

# Past exams

## Answer keys (worked out from the slides; check anything that surprises you)
**Jan 2026:** 1d 2c 3b 4c 5a 6b 7d 8b 9a 10c 11a 12c 13d 14a 15a 16a 17c 18c 19b 20d
**Apr 2026 re-exam:** 1c 2b 3c 4d 5a 6b 7a 8d 9b 10c 11d 12b 13d 14c 15c 16a 17d 18b 19c 20b

## Later topics on the exams (these get detailed notes once you upload those lectures)

- **CNNs:** an MLP flattens an image into a long vector, which discards which pixels are next to each other, and needs a huge number of weights. A convolution slides one small filter across the whole image. **Shared weights** mean far fewer parameters, and a feature like an edge is detected wherever it appears. Output size = (n − f + 2p)/s + 1, so a 5×5 filter on a 32×32 image gives 28×28.
- **RNNs → attention → Transformers:** an RNN squeezes the whole past into one hidden state, so distant information fades. Attention lets each step look directly at any earlier state. Transformers drop recurrence entirely, so all positions are processed **in parallel**. In self-attention, Q, K and V all come from the same sequence. Multiple heads let the model attend to several kinds of relationships at once.
- **Feature stores and pipelines:** compute features once and reuse them identically in training and inference, which avoids training/serving skew. Time travel retrieves feature values *as they were* at a past time, so training data is reproducible. A feature view joins and selects features. Precomputed features cost storage but are fast. On-demand features save storage but add latency. Use event time, not processing time, so late and out-of-order events land in the correct window.
- **Testing and monitoring:** feature unit tests require feature logic written as independent, testable functions. Contract tests check schema, ranges and meaning. Lineage traces data → features → model → deployment. Guardrails filter harmful or invalid outputs at runtime.
- **Drift:** feature (covariate) drift is a change in the input distribution while the task stays the same, so **retrain** on fresh data. Concept drift is a change in the X→y relationship itself, so **redesign**.
- **Hyperparameter tuning:** successive halving (SHA) gives many configurations a small budget, then keeps the best half and doubles their budget. But you must choose many configurations with little budget each, or few with more. HyperBand runs SHA with **several such trade-offs** and covers both.
- **Distributed training:** a parameter server is a central node that every worker sends gradients to, which makes it a bottleneck. AllReduce spreads the communication among the workers themselves. Data parallelism copies the model to each device, each with different data. **Model parallelism** splits the model itself across devices, which you need when it doesn't fit on one.

## Study routine

1. Read one lecture section, then explain it out loud without notes. Then take the L1–L4 quiz closed-book and aim for 15 or more out of 16.
2. For each new lecture, upload the slides and I'll add detailed notes in this format.
3. Every week, take both past exams closed-book and aim for 19 or more out of 20.
4. In the last 2 weeks, only do mixed practice, with extra time on the weakest block (MLOps is the biggest).
