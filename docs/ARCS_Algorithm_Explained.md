# ARCS: Autoregressive Circuit Synthesis — Full Algorithm Explanation

> A comprehensive guide to understanding every component of the ARCS system,
> written for someone who may not have a background in machine learning or circuit design.

---

## Table of Contents

1. [The Big Picture](#1-the-big-picture)
2. [Key Terms Glossary](#2-key-terms-glossary)
3. [Step 1: Tokenization — Turning Circuits Into Numbers](#3-step-1-tokenization)
4. [Step 2: Embeddings — Giving Meaning to Numbers](#4-step-2-embeddings)
5. [Step 3: RWPE — Teaching the Model About Circuit Structure](#5-step-3-rwpe)
6. [Step 4: The GraphTransformer — The Brain](#6-step-4-the-graphtransformer)
7. [Step 5: Two-Head Output — Structure vs. Values](#7-step-5-two-head-output)
8. [Step 6: Constrained Generation — Keeping It Valid](#8-step-6-constrained-generation)
9. [Step 7: Training — How the Model Learns](#9-step-7-training)
10. [Step 8: Inference — Using the Trained Model](#10-step-8-inference)
11. [Putting It All Together — A Complete Example](#11-putting-it-all-together)
12. [Figure Guide](#12-figure-guide)

---

## 1. The Big Picture

ARCS solves this problem: **given a set of electrical specifications (like "I need 12V in, 5V out, 1A current"), automatically design a circuit that meets those specs.**

Traditional circuit design requires an experienced engineer to:
1. Choose a circuit topology (e.g., a Buck converter for stepping down voltage)
2. Select component types (resistors, capacitors, inductors, transistors)
3. Calculate component values (4.7kΩ, 22µF, 100µH, etc.)
4. Verify the design works through simulation

ARCS automates steps 2–3 using a neural network. It treats circuit design as a **language problem** — just like GPT predicts the next word in a sentence, ARCS predicts the next component in a circuit.

### The Pipeline (Figure 1)

```
Design Spec → Tokenizer → GraphTransformer → Constrained Decoder → SPICE Netlist
```

- **Input:** "Buck converter, Vin=12V, Vout=5V, Iout=1A, Fsw=100kHz"
- **Output:** A complete circuit description with specific component values

---

## 2. Key Terms Glossary

### Machine Learning Terms

| Term | What It Means |
|------|--------------|
| **Token** | The smallest unit the model works with. Like how GPT treats words/subwords as tokens, ARCS treats component types and values as tokens. A token is just an integer (e.g., token #25 = "TOPO_BUCK"). |
| **Vocabulary** | The complete set of all possible tokens. ARCS has 686 tokens total. |
| **Embedding** | A way to represent a token as a list of numbers (a "vector"). Token #25 doesn't mean anything to math — but a 256-number vector like [0.3, -0.1, 0.7, ...] can encode rich meaning. The model learns these representations during training. |
| **Autoregressive** | Generating output one token at a time, left to right. Each new token depends on all previous tokens. Like writing a sentence word by word — each word choice depends on what you've already written. |
| **Transformer** | A neural network architecture (invented in 2017) that uses "attention" to process sequences. It's the architecture behind GPT, BERT, and now ARCS. |
| **Attention** | A mechanism that lets each position in the sequence look at every previous position and decide which ones are most relevant. Like reading a sentence and paying more attention to the subject when predicting the verb. |
| **Self-Attention** | When the sequence attends to itself (as opposed to cross-attention between two different sequences). |
| **Causal Masking** | Preventing future tokens from being visible during generation. Position 5 can only see positions 1–4, not 6+. This is what makes it "autoregressive." |
| **Softmax** | A function that converts a list of numbers into probabilities that sum to 1. E.g., [2.0, 1.0, 0.5] → [0.59, 0.24, 0.17]. Used to pick the most likely next token. |
| **Cross-Entropy Loss** | A way to measure how wrong the model's predictions are. If the correct token is #42 and the model assigns 90% probability to #42, the loss is low. If it assigns 1%, the loss is high. |
| **Perplexity** | 2^(cross-entropy loss). Intuitively, it's "how many tokens is the model choosing between on average." A perplexity of 2.4 means the model is about as uncertain as choosing between 2–3 options. Lower = better. |
| **Gradient Descent** | The algorithm that adjusts the model's parameters to reduce the loss. Like rolling a ball downhill — it finds the direction that reduces error and takes a small step. |
| **Learning Rate** | How big each gradient descent step is. Too large = overshooting. Too small = takes forever. ARCS uses 3×10⁻⁴ as the peak learning rate. |
| **AdamW** | A popular optimizer (variant of gradient descent) that adapts the learning rate per-parameter and includes weight decay (regularization). |
| **Weight Decay** | A regularization technique that gently pushes model weights toward zero, preventing overfitting. ARCS uses 0.1. |
| **Epoch** | One complete pass through the entire training dataset. ARCS trains for 100 epochs. |
| **Batch Size** | How many training examples the model processes at once before updating its weights. ARCS uses 64. |
| **Overfitting** | When the model memorizes the training data but fails on new data. The "generalization gap" (training loss much lower than validation loss) indicates overfitting. |
| **Dropout** | A regularization technique: during training, randomly set 10% of values to zero. This prevents the model from relying too heavily on any single feature. |
| **Residual Connection** | Adding the input of a layer back to its output: output = layer(x) + x. This helps information flow through deep networks and makes training more stable. |
| **RMSNorm** | A normalization technique that scales vectors to have consistent magnitude. Helps training stability. Like ensuring all numbers are roughly the same scale. |
| **Weight Tying** | Sharing the same parameters between the input embedding and the output prediction layer. Since both map between tokens and vectors, they can share weights, reducing model size. |
| **MLP** | Multi-Layer Perceptron — a simple neural network of stacked linear layers with nonlinear activations between them. |
| **GELU / SiLU** | Activation functions (nonlinearities). They introduce curves into what would otherwise be straight-line math, allowing the network to learn complex patterns. SiLU(x) = x × sigmoid(x). |
| **SwiGLU** | A gated feed-forward network: instead of one linear layer, use two in parallel and multiply their outputs. One acts as a "gate" controlling information flow. More expressive than standard FFN. |
| **Parameters** | The learnable numbers in the model. ARCS has 6,829,296 parameters (~6.8 million). For comparison, GPT-2 has 117 million, GPT-3 has 175 billion. |

### Circuit Design Terms

| Term | What It Means |
|------|--------------|
| **Topology** | The circuit architecture/type. Like choosing between a sedan, SUV, or truck before picking the specific model. Examples: Buck converter (steps voltage down), Boost converter (steps voltage up), Sallen-Key filter (filters frequencies). |
| **Component** | A physical part in the circuit: resistor (R), capacitor (C), inductor (L), MOSFET (transistor), op-amp, transformer, diode. |
| **Component Value** | The specific electrical property: a resistor's resistance (4.7kΩ), a capacitor's capacitance (22µF), an inductor's inductance (100µH). |
| **SPICE** | Simulation Program with Integrated Circuit Emphasis — the industry-standard circuit simulator since 1973. A SPICE netlist is a text file describing the circuit for simulation. |
| **Netlist** | A text description of a circuit listing all components, their values, and their connections (which pins connect to which wires/nodes). |
| **Vin, Vout** | Input voltage, output voltage. For a Buck converter: Vin=12V, Vout=5V means it converts 12V down to 5V. |
| **Iout** | Output current. How much current the circuit needs to deliver. |
| **Fsw** | Switching frequency. How fast the transistors switch on/off in a power converter (typically 50kHz–1MHz). |
| **ESR** | Equivalent Series Resistance — real capacitors aren't perfect; they have a small parasitic resistance. |
| **E-series** | Standard sets of preferred component values used in industry (E12, E24, E96). Components only come in certain values — you can't buy a 3.14kΩ resistor, but you can buy 3.3kΩ. |

### ARCS-Specific Terms

| Term | What It Means |
|------|--------------|
| **RWPE** | Random-Walk Positional Encoding. A way to tell the model about the circuit's graph structure (which components connect to which). Explained in detail in [Section 5](#5-step-3-rwpe). |
| **Graph Bias** | Extra attention weights added for tokens that represent connected components. If an inductor connects to a capacitor, they attend to each other more strongly. |
| **Two-Head Output** | The model has two separate prediction heads: one for structural tokens (what type of component comes next) and one for value tokens (what value that component should have). |
| **Constrained Decoder** | A rule engine that prevents the model from generating invalid circuits. It masks out impossible tokens at each step. |
| **Grammar FSM** | Finite State Machine for grammar. A set of rules about what token types can follow what. Components must alternate with values; you can't have two component types in a row. |

---

## 3. Step 1: Tokenization — Turning Circuits Into Numbers

Neural networks only understand numbers. So we need to convert circuit descriptions into sequences of integers.

### The 686-Token Vocabulary (Figure 5, right panel)

The vocabulary is divided into blocks:

```
Tokens 0–4:     Special tokens (START, END, SEP, PAD, INVALID)
Tokens 5–24:    Component types (COMP_RESISTOR, COMP_CAPACITOR, COMP_INDUCTOR, ...)
Tokens 25–44:   Topology types (TOPO_BUCK, TOPO_BOOST, TOPO_FLYBACK, ...)
Tokens 45–64:   Specification keys (SPEC_VIN, SPEC_VOUT, SPEC_IOUT, ...)
Tokens 65–185:  Pin and net tokens (for describing connectivity — future use)
Tokens 186–685: Value bins (500 bins spanning 1 picofarad to 1 megaohm)
```

### How Values Are Discretized (Figure 5, left panel)

Real component values span an enormous range:
- A capacitor might be 1 picofarad (10⁻¹²) or 1000 microfarads (10⁻³)
- A resistor might be 0.1 ohms or 10 megaohms (10⁷)

To handle this, we divide the range [10⁻¹², 10⁶] into **500 bins on a logarithmic scale**. This means:
- Bin 0 covers ~1pF to ~1.5pF
- Bin 250 covers ~0.1 to ~0.15
- Bin 499 covers ~700kΩ to ~1MΩ

**Why logarithmic?** Because in electronics, the difference between 100Ω and 101Ω doesn't matter, but the difference between 100Ω and 1kΩ does. Log-spacing gives equal importance to each order of magnitude.

**Example encoding:**
```
22µH = 2.2×10⁻⁵
log₁₀(2.2×10⁻⁵) = -4.66
bin = floor((-4.66 + 12) / 18 × 500) = floor(204) = 204
token ID = 186 + 204 = 390
```

**Decoding** reverses this: take the geometric mean of the bin edges to recover an approximate value.

### The Token Sequence Format

Every circuit is encoded as a flat sequence:

```
START → TOPO_BUCK → SEP → SPEC_VIN → 12.0 → SPEC_VOUT → 5.0 → SPEC_IOUT → 1.0 → SPEC_FSW → 100k → SEP → COMP_RESISTOR → 4.7k → COMP_INDUCTOR → 22µ → COMP_CAPACITOR → 100µ → COMP_MOSFET_N → 0.01 → END
```

Breaking this down:
1. **START** — signals the beginning
2. **TOPO_BUCK** — declares the topology
3. **SEP** — separator between sections
4. **Spec section:** alternating spec-key/value pairs (what the circuit should do)
5. **SEP** — separator
6. **Component section:** alternating component-type/value pairs (what parts to use)
7. **END** — signals completion

The **spec section** is the input (what we want), and the **component section** is the output (what the model generates).

---

## 4. Step 2: Embeddings — Giving Meaning to Numbers

A token ID like 25 is just an arbitrary number. To make it useful, we convert it into a **256-dimensional vector** (a list of 256 numbers) that encodes its meaning. ARCS sums **four different embeddings** for each token:

### 4a. Token Embedding (686 × 256)

Each of the 686 tokens gets its own unique 256-number vector. Initially random, these vectors are learned during training. After training, similar tokens (like COMP_RESISTOR and COMP_CAPACITOR) end up with similar vectors, while dissimilar tokens (like SPEC_VIN and END) have very different vectors.

### 4b. Position Embedding (128 × 256)

Position 1, position 2, ..., position 128 each get their own vector. This tells the model *where* in the sequence a token appears. Without this, the model couldn't distinguish "R → 4.7k → C → 100µ" from "C → 100µ → R → 4.7k" — the tokens would look the same regardless of order.

### 4c. Type Embedding (7 × 256)

There are 7 token categories: SPECIAL, COMPONENT, VALUE, PIN, CONNECTION, SPEC, TOPOLOGY. This gives the model a high-level understanding of what role each token plays, separate from its specific identity.

### 4d. RWPE Embedding (8 → 256)

This encodes graph structure information — which components are electrically connected. Explained in detail next.

### Combining Them

All four embeddings are added element-wise:

```
final_embedding[i] = token_emb[token_id] + pos_emb[position] + type_emb[token_type] + rwpe_proj(rwpe_features)
```

Then dropout (randomly zeroing 10% of values) is applied for regularization.

---

## 5. Step 3: RWPE — Teaching the Model About Circuit Structure

This is one of ARCS's key innovations. Standard transformers only know about **sequence position** (token 1, token 2, ...), but circuits have **graph structure** (the inductor connects to the MOSFET, which connects to the capacitor). RWPE encodes this structure.

### The Problem

Consider a Buck converter with 4 components: Inductor (L), Capacitor (C), ESR (a parasitic resistance), and MOSFET (M). Their connections form a graph:

```
    L
   / \
  M   C
      |
     ESR
```

L connects to both M and C (degree 2). C connects to L and ESR (degree 2). M connects only to L (degree 1). ESR connects only to C (degree 1).

The model needs to know this structure to make good predictions — the inductor's value depends on what it's connected to.

### The Solution: Random Walks (Figure 3)

Imagine an ant walking randomly on the circuit graph. At each step, it moves to a randomly chosen neighbor:
- From L, the ant goes to M or C (50/50 chance each)
- From M, the ant can only go to L (100% chance)
- From ESR, the ant can only go to C (100% chance)

**RWPE asks: "If the ant starts at node X, what's the probability it returns to X after k steps?"**

This gives each node a unique "fingerprint" based on its structural role:
- **Hub nodes** (L, C with degree 2): the ant has multiple paths, so return probability builds up gradually
- **Leaf nodes** (M, ESR with degree 1): the ant must return via the same edge, so return probability at k=2 is 100%

### The Math

1. **Adjacency matrix A:** A 4×4 matrix where A[i][j] = 1 if nodes i and j are connected

```
    L  C  R  M
L [ 0  1  0  1 ]
C [ 1  0  1  0 ]
R [ 0  1  0  0 ]
M [ 1  0  0  0 ]
```

2. **Degree matrix D:** Diagonal matrix with each node's connection count

```
D = diag(2, 2, 1, 1)
```

3. **Transition matrix T = D⁻¹A:** Normalize each row by the node's degree. Now T[i][j] = probability of walking from i to j in one step.

```
      L    C    R    M
L [ 0.0  0.5  0.0  0.5 ]    ← from L: 50% to C, 50% to M
C [ 0.5  0.0  0.5  0.0 ]    ← from C: 50% to L, 50% to ESR
R [ 0.0  1.0  0.0  0.0 ]    ← from ESR: 100% to C
M [ 1.0  0.0  0.0  0.0 ]    ← from M: 100% to L
```

4. **Powers T^k:** T^k[i][i] = probability of returning to node i after exactly k steps.

```
RWPE(L)   = [T¹ₗₗ, T²ₗₗ, T³ₗₗ, ..., T⁸ₗₗ] = [0.00, 0.50, 0.00, 0.38, 0.00, 0.34, ...]
RWPE(ESR) = [T¹ᵣᵣ, T²ᵣᵣ, T³ᵣᵣ, ..., T⁸ᵣᵣ] = [0.00, 1.00, 0.00, 0.50, 0.00, 0.50, ...]
```

Notice ESR has T²=1.00 (it *must* return after 2 steps — go to C, come back) while L has T²=0.50 (it returns only if the ant goes to M or C and comes back, which happens half the time). These different profiles encode different structural roles.

5. **Project to embedding space:** The 8-dimensional RWPE vector gets projected to 256 dimensions through a small neural network:

```
RWPE features (8 dims) → Linear layer (8→64) → GELU activation → Linear layer (64→256) → added to token embedding
```

This projection has only 17,216 parameters (0.25% of the total model) — a tiny cost for significant structural awareness.

### Why K=8 Walk Lengths?

Short walks (k=1,2) capture local structure (immediate neighbors). Longer walks (k=5,6,7,8) capture global structure (is this node in a cycle? how far from the graph center?). Eight walk lengths provide a good balance.

### Precomputation

RWPE is computed once per topology at import time (not during training). Since ARCS supports 16 fixed topologies, we precompute and cache all 16 RWPE matrices. During training/inference, we just look up the right one.

---

## 6. Step 4: The GraphTransformer — The Brain

This is the core neural network. It takes the embedded token sequence and produces rich representations that capture the meaning and context of each token. (Figure 2)

### Architecture Overview

```
Input embeddings (256-dim per token)
    ↓
[GraphTransformerBlock] × 6    ← repeat the block 6 times
    ↓
RMSNorm
    ↓
Output heads (Structure + Value)
```

### Inside Each GraphTransformerBlock

Each block has two sub-layers with residual connections:

#### Sub-layer 1: Graph-Aware Causal Self-Attention

This is where the model decides which previous tokens to "pay attention to" when processing each position.

**Standard attention** works like this:
1. For each token, compute three vectors: Query (Q), Key (K), Value (V) — each 256-dimensional, split across 4 heads of 64 dimensions each
2. The "attention score" between token i and token j = dot product of Q_i and K_j, divided by √64 (for numerical stability)
3. Apply softmax to get attention weights (probabilities summing to 1)
4. The output for token i = weighted sum of all Value vectors, weighted by attention scores

**ARCS adds two graph-aware biases** to the attention scores:

- **Adjacency bias:** For each attention head, learn a scalar weight. Add this weight to the attention score whenever two tokens represent connected components in the circuit graph. This makes the model naturally pay more attention to electrically connected components.

```
attention_score[i][j] += b_adj[head] × A_graph[i][j]
```

- **Edge-type bias:** Learn different biases for different component-type pairs. A resistor-capacitor connection gets a different bias than an inductor-MOSFET connection. There are 17 edge-type buckets × 4 attention heads = 68 learned biases.

```
attention_score[i][j] += b_edge[edge_type(i,j)][head]
```

**Causal masking** is also applied: token i can only attend to tokens 1, 2, ..., i (not future tokens). This is essential for autoregressive generation.

The final attention formula:

$$\text{Attention} = \text{softmax}\left(\frac{QK^\top}{\sqrt{64}} + b_{\text{adj}} \cdot A_{\text{graph}} + b_{\text{edge}}(E) + M_{\text{causal}}\right) V$$

Where $M_{\text{causal}}$ is -∞ for future positions (making their softmax weight exactly 0).

#### Sub-layer 2: SwiGLU Feed-Forward Network

After attention, each token's representation is processed independently through a feed-forward network:

```
input x (256-dim)
    ↓
Two parallel linear layers:
    W₁x → SiLU activation    (256 → 1024)
    W₃x                       (256 → 1024)
    ↓
Element-wise multiply: SiLU(W₁x) ⊙ W₃x    (1024-dim)
    ↓
W₂ → output    (1024 → 256)
    ↓
Dropout (10%)
```

The "gate" (W₃) learns to control which features pass through, making this more selective than a standard feed-forward layer.

#### Residual Connections

Each sub-layer has a residual connection: `output = sublayer(x) + x`. This means information can flow directly through the network without being transformed, helping with training deep networks.

#### Normalization

RMSNorm is applied before each sub-layer (called "pre-norm"), not after. This is more stable than the original transformer's post-norm.

### Parameter Count Breakdown

| Component | Parameters | % of Total |
|-----------|-----------|------------|
| Token embedding (686×256) | 175,616 | 2.6% |
| Position embedding (128×256) | 32,768 | 0.5% |
| Type embedding (7×256) | 1,792 | 0.0% |
| RWPE projection (8→64→256) | 17,216 | 0.3% |
| Attention QKV (256→768) × 6 layers | 1,179,648 | 17.3% |
| Attention output (256→256) × 6 | 393,216 | 5.8% |
| Graph bias params × 6 | 432 | 0.0% |
| FFN SwiGLU (256→1024→256) × 6 | 4,718,592 | 69.1% |
| Norms × 6 | 3,328 | 0.0% |
| Value projection MLP | 131,072 | 1.9% |
| Value head (256→686) | 175,616 | 2.6% |
| **Total** | **6,829,296** | **100%** |

Note: The structure head reuses the token embedding weights (weight tying), so it adds no extra parameters.

The FFN layers dominate at 69% — this is typical for transformers.

---

## 7. Step 5: Two-Head Output — Structure vs. Values

After the 6 transformer blocks, the model needs to predict the next token. But there's an insight: **predicting "the next component is a resistor" is a very different task from predicting "its resistance is 4.7kΩ."**

### Why Two Heads?

- **Structural prediction** (what type of component): This is about understanding circuit topology and what components are needed. It's a categorical choice from ~65 options.
- **Value prediction** (what value): This is about understanding electrical engineering constraints — calculating the right inductance for a given switching frequency and current. It's choosing from 500 finely-graded numerical bins.

A single shared output layer would be forced to compromise between these very different tasks. Two separate heads let each specialize.

### Structure Head

```
hidden state (256-dim) → Linear(256→686) → logits
```

The weight matrix is **tied** (shared) with the token embedding layer. This is possible because both map between the same token space and vector space, just in opposite directions. Weight tying saves 175,616 parameters and acts as a regularizer.

### Value Head

```
hidden state (256-dim) → MLP(256→256→256, with SiLU) + residual → Linear(256→686) → logits
```

The value head has an extra MLP with a **residual connection**: `h_val = MLP(h) + h`. This gives the value head its own learned transformation before the final projection. The extra MLP has independent weights (not tied).

### Routing

During training, a **value mask** determines which positions use which head:
- If the target token is a value (VAL_*): use value head logits
- Otherwise: use structure head logits
- Both heads are trained simultaneously via the combined loss

During generation:
- If the last token emitted was a component type (COMP_*): the next token is a value → use value head
- Otherwise: use structure head

---

## 8. Step 6: Constrained Generation — Keeping It Valid

The model is powerful but not perfect. Without constraints, it might generate nonsense like "COMP_RESISTOR COMP_CAPACITOR COMP_INDUCTOR" (three component types in a row with no values) or assign a 1MΩ resistance to a MOSFET gate (physically meaningless).

### The Grammar State Machine (Figure 4, top)

A finite state machine (FSM) enforces valid circuit grammar:

```
PREFIX → EXPECT_COMP ↔ EXPECT_VAL → DONE
```

**States:**

1. **PREFIX:** The spec tokens are given as input. Not generated by the model.

2. **EXPECT_COMP:** The model must output a component type token. All other tokens (values, specs, etc.) are masked to -∞ in the logits (making their probability effectively 0 after softmax). After ≥2 components have been placed, the END token is also allowed.

3. **EXPECT_VAL:** The model must output a value token. Only VAL_* tokens are unmasked.

4. **DONE:** END was emitted. Generation stops.

### Three Constraint Levels (Figure 4, bottom)

The FSM supports three levels of strictness:

**Level 1 — Grammar Only:**
- Enforces alternating COMP → VAL pattern
- END only allowed after ≥2 components
- No other restrictions

**Level 2 — Topology-Aware:**
- Everything from Level 1, plus:
- Only component types expected by the chosen topology are allowed
- Example: A Buck converter expects [INDUCTOR, CAPACITOR, RESISTOR, MOSFET_N]. The model cannot generate COMP_OPAMP or COMP_TRANSFORMER for a Buck.

**Level 3 — Full Physical Constraints:**
- Everything from Level 2, plus:
- Value tokens are restricted to physically valid ranges per component per topology
- Example: For a Buck converter's inductance, valid range might be [1µH, 10mH]. Value tokens outside this range are masked.
- These ranges come from `ComponentBounds` definitions in the SPICE template library.

### How Masking Works

At each generation step:
1. The model produces raw logits (686 numbers, one per token)
2. The constraint engine produces a mask (686 numbers: 0.0 for valid tokens, -∞ for invalid)
3. The mask is added to the logits: `constrained_logits = logits + mask`
4. Softmax is applied to get probabilities
5. A token is sampled from these probabilities

Since e^(-∞) = 0, masked tokens get exactly zero probability. The model can only choose from valid tokens.

---

## 9. Step 7: Training — How the Model Learns

### Training Data

ARCS is trained on 53,000 circuit samples across 16 topologies, stored as JSONL files. Each sample contains:
- Topology type
- Specification values (Vin, Vout, etc.)
- Component types and values

These circuits were generated by sampling random parameters within valid bounds and verifying them through SPICE simulation.

### Data Augmentation (5×)

To increase the dataset from 53K to 161K samples:
- **Eulerian augmentation:** Reorder components in ways that maintain validity (like shuffling the order you list parts)
- **Shuffle augmentation:** Random permutation of component order within the output section

This teaches the model that component order doesn't matter — only the types and values matter.

### Loss Function

The loss is weighted cross-entropy:

$$\mathcal{L} = \frac{\sum_i w_i \cdot \text{CE}(\text{logits}_i, \text{target}_i)}{\sum_i w_i}$$

Where:
- CE = cross-entropy between the model's predicted probabilities and the correct token
- $w_i = 1$ for structural tokens
- $w_i = 5$ for value tokens (the `value_weight` hyperparameter)

**Why weight values 5× more?** Value prediction is harder and more important. Getting the topology and component types right is easier (limited choices), but getting component values right requires precise numerical understanding. The 5× weight tells the optimizer to prioritize getting values right.

### Optimizer Settings

| Setting | Value | Why |
|---------|-------|-----|
| Optimizer | AdamW | Adaptive learning rates + weight decay |
| Peak learning rate | 3×10⁻⁴ | Standard for small transformers |
| Weight decay | 0.1 | Moderate regularization |
| β₁, β₂ | 0.9, 0.95 | Momentum for gradient estimates |
| Gradient clipping | 1.0 | Prevents exploding gradients |
| Batch size | 64 | Balance between speed and gradient noise |

### Learning Rate Schedule

The learning rate isn't constant. It follows a "warmup + cosine decay" schedule:

1. **Warmup (epochs 1–5):** LR linearly increases from 0 to 3×10⁻⁴. This prevents early training instability when the model's parameters are still random.

2. **Cosine decay (epochs 6–100):** LR smoothly decreases following a cosine curve, dropping to 10% of peak (3×10⁻⁵). This helps the model converge to a stable solution — large steps early for fast progress, tiny steps later for fine-tuning.

### Training Metrics

After each epoch, the model is evaluated on a held-out validation set:

- **Validation loss:** Cross-entropy on data the model hasn't trained on. Should decrease and then plateau.
- **Overall accuracy:** % of tokens predicted correctly
- **Structural accuracy:** % of structural tokens (COMP_*, TOPO_*, etc.) correct
- **Value accuracy:** % of value tokens correct (harder — consistently lower than structural)
- **Perplexity:** Exponential of the loss. "How many tokens is the model confused between?"

### What We Observed (Training Curves)

Over 100 epochs on the GraphTransformer with RWPE:

| Metric | Epoch 1 | Epoch 50 | Epoch 80 (best) | Epoch 100 |
|--------|---------|----------|-----------------|-----------|
| Val loss | ~2.10 | ~0.90 | **0.8718** | 0.873 |
| Perplexity | ~8.2 | ~2.5 | **2.39** | 2.39 |
| Structural acc | ~65% | ~90% | **91.1%** | 91.1% |
| Value acc | ~30% | ~72% | **72.4%** | 72.3% |
| Learning rate | 6×10⁻⁵ | 2×10⁻⁴ | 3.2×10⁻⁵ | 3.0×10⁻⁵ |

The model learns fast in the first 30 epochs, then gradually improves. After epoch 80, improvement stops (the model has converged). The best checkpoint is saved automatically at epoch 80.

---

## 10. Step 8: Inference — Using the Trained Model

Once trained, here's how the model generates a circuit:

### Input Preparation

1. User provides specs: "Buck converter, 12V→5V, 1A, 100kHz"
2. This is encoded as a prefix sequence:

```
[START, TOPO_BUCK, SEP, SPEC_VIN, VAL_310, SPEC_VOUT, VAL_283, SPEC_IOUT, VAL_333, SPEC_FSW, VAL_472, SEP]
```

### Autoregressive Generation

Starting from the prefix, generate one token at a time:

```
Step 1: Give prefix to model → model predicts logits for position 13
        → apply constraint mask (only COMP_* tokens allowed)
        → softmax → sample → COMP_INDUCTOR

Step 2: Append COMP_INDUCTOR → predict logits for position 14
        → apply constraint mask (only VAL_* tokens allowed, within inductor range)
        → softmax → sample → VAL_265 (≈ 22µH)

Step 3: Append VAL_265 → predict logits for position 15
        → apply constraint mask (COMP_* or END allowed)
        → softmax → sample → COMP_CAPACITOR

Step 4: Append COMP_CAPACITOR → predict position 16
        → constrain to capacitor value range
        → sample → VAL_350 (≈ 100µF)

... continue until END is generated ...
```

### Temperature and Sampling

When sampling from the probability distribution, we can control randomness:
- **Temperature = 0** (or greedy): always pick the most likely token. Deterministic but potentially boring.
- **Temperature = 1**: sample according to model probabilities. More diverse.
- **Top-k sampling**: only consider the k most likely tokens.
- **Top-p (nucleus) sampling**: only consider tokens whose cumulative probability ≤ p.

### Best-of-N

Generate N candidate circuits, score each with the learned reward model, and return the best one. This is a simple form of "inference-time compute scaling" — spending more compute at inference to get better results.

---

## 11. Putting It All Together — A Complete Example

**User request:** "Design a Buck converter: 12V input, 5V output, 1A, 100kHz switching frequency"

**Step 1 — Tokenize the spec:**
```
[1, 25, 3, 45, 310, 46, 283, 47, 333, 51, 472, 3]
 ↑   ↑  ↑   ↑   ↑   ↑   ↑   ↑   ↑   ↑   ↑  ↑
 S  BUCK SEP VIN 12V VOUT 5V IOUT 1A FSW 100k SEP
```

**Step 2 — Look up RWPE:**
Buck topology has 4 nodes (L, C, ESR, M). Precomputed RWPE vectors are loaded:
```
RWPE(L)   = [0.00, 0.50, 0.00, 0.38, 0.00, 0.34, 0.00, 0.32]  (degree-2 hub)
RWPE(C)   = [0.00, 0.50, 0.00, 0.38, 0.00, 0.34, 0.00, 0.32]  (degree-2 hub)
RWPE(ESR) = [0.00, 1.00, 0.00, 0.50, 0.00, 0.50, 0.00, 0.38]  (degree-1 leaf)
RWPE(M)   = [0.00, 1.00, 0.00, 0.50, 0.00, 0.50, 0.00, 0.38]  (degree-1 leaf)
```

**Step 3 — Embed each token:**
```
embedding[i] = tok_emb[token_id[i]] + pos_emb[i] + type_emb[type[i]] + rwpe_proj(rwpe[i])
```

**Step 4 — Process through 6 transformer blocks:**
Each block:
- RMSNorm → Graph-Aware Attention (attend to connected components more) → residual
- RMSNorm → SwiGLU FFN → residual

After 6 blocks, each token has a rich 256-dimensional representation encoding:
- What token it is
- Where it is in the sequence
- What type of token it is
- What it's connected to in the circuit graph
- Context from all previous tokens (via attention)

**Step 5 — Generate component section:**
```
Position 13 → Structure Head → COMP_INDUCTOR (L)
Position 14 → Value Head → VAL_265 → 22µH
Position 15 → Structure Head → COMP_CAPACITOR (C)
Position 16 → Value Head → VAL_350 → 100µF
Position 17 → Structure Head → COMP_RESISTOR (ESR)
Position 18 → Value Head → VAL_186 → 0.01Ω
Position 19 → Structure Head → COMP_MOSFET_N (M)
Position 20 → Value Head → VAL_225 → 0.005Ω (Rdson)
Position 21 → Structure Head → END
```

**Step 6 — Decode tokens to SPICE netlist:**
```spice
* Buck Converter - ARCS Generated
.param Vin=12 Vout=5 Fsw=100k

L1  sw_node  out  22u
C1  out  gnd  100u
R_esr  out  cap_top  0.01
M1  vin  gate  sw_node  sw_node  NMOS W=10u L=0.18u

.tran 100u
.measure TRAN vout_avg AVG V(out) FROM=50u TO=100u
.end
```

**Step 7 — Simulate and verify:**
The generated netlist can be run through SPICE to verify the circuit meets specs.

---

## 12. Figure Guide

### Figure 1: End-to-End Pipeline (`1_pipeline.png`)
- **What it shows:** The complete flow from user specs to SPICE output
- **Read left to right:** Each box is a processing stage
- **Bottom row:** An actual token sequence for a Buck converter example
- **Top gray box:** Training loop details (loss function, optimizer)
- **Key insight:** The spec section (blue tokens) is input; the component section (green/pink tokens) is generated

### Figure 2: GraphTransformer Architecture (`2_transformer_block.png`)
- **What it shows:** The internal wiring of the neural network
- **Read bottom to top:** Data flows upward from embeddings through transformer blocks to output heads
- **Four embedding boxes at bottom:** Token + Position + Type + RWPE, all summed
- **Dashed box in middle:** One GraphTransformerBlock, repeated 6 times
- **Key detail:** The attention formula inside the block — note the graph bias terms (adj_bias · A and edge_type)
- **Red "+" symbols:** Residual connections (skip connections that help training)
- **Top:** Two output heads splitting into Structure (green) and Value (pink)

### Figure 3: RWPE Computation (`3_rwpe.png`)
- **Left panel:** Buck converter graph with 4 nodes and their degrees
- **Middle panel:** Adjacency matrix A and transition matrix T = D⁻¹A, with example RWPE vectors computed for each node
- **Right panel:** The projection pipeline (8 → 64 → 256 dimensions)
- **Key insight:** Degree-2 nodes (L, C) have different return probabilities than degree-1 nodes (ESR, M), giving each a unique structural fingerprint

### Figure 4: Constrained Generation (`4_constrained_gen.png`)
- **Top:** State machine with 4 states connected by arrows showing valid transitions
- **Key detail:** EXPECT_COMP and EXPECT_VAL alternate — you can't have two components in a row
- **Bottom:** Three nesting constraint levels, each adding more restrictions
- **Key insight:** Level 3 (FULL) ensures generated values are physically realistic

### Figure 5: Value Tokenization (`5_tokenization.png`)
- **Left panel:** Bar chart showing log-spaced bins. X-axis is physical value (log scale), Y-axis is bin width
- **Right panel:** Vocabulary layout with color-coded blocks for each token category
- **Bottom:** Example encode/decode roundtrip showing how 22µH maps to bin 222 and back
- **Key insight:** Log-spacing gives equal resolution to each order of magnitude — important because component values span 18 orders of magnitude

### Training Curves (`training_curves.png`)
- **Top-left:** Train (blue) and val (red) loss over 100 epochs. Best val loss at epoch 80.
- **Top-middle:** Perplexity — dropped from ~8 to ~2.4
- **Top-right:** Accuracy breakdown — structural accuracy (91%) is much higher than value accuracy (72%)
- **Bottom-left:** Learning rate schedule — warmup then cosine decay
- **Bottom-middle:** Generalization gap (val - train). Growing gap after epoch 60 indicates mild overfitting.
- **Bottom-right:** Time per epoch (~22 minutes on M3 MacBook Air)

---

*Generated for the ARCS project (ICCAD 2026). All figures are in `results/algorithm_figures/` and `results/training_curves.png`.*
