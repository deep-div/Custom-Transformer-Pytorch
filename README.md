# 🧠 Transformer-from-Scratch (PyTorch)

A complete implementation of the Transformer architecture from scratch using PyTorch. This project aims to help researchers, students, and enthusiasts understand the inner workings of the Transformer model without relying on high-level abstractions from libraries like HuggingFace or Fairseq.

## 📌 Features

* Custom implementation of:

  * Positional Encoding
  * Scaled Dot-Product Attention
  * Multi-Head Attention
  * Feedforward Network
  * Encoder & Decoder Stacks
  * Attention Masking
* Easy to read and modular code
* Designed for educational clarity and flexibility


# 🔹 Understanding Self-Attention in Multi-Head Attention


## **Step 1: Tokenization and Embedding**  
Before attention is applied, each word in the sentence is **tokenized**, then converted into **embeddings**, concatenated with its **positional encoding**, and finally represented as a numerical vector (**embedding**).

- In the *"Attention Is All You Need"* paper, `d_model = 512`. Each word is represented as a **1D vector of 512 dimensions (512D)**.

### **Example Sentence:**
> **"Hi how are you"**

Each word is converted into a vector:

- **"Hi"** → `[0.1, 0.3, 0.7, ...]`  (Size: `d_model`)
- **"how"** → `[0.2, 0.6, 0.5, ...]`  (Size: `d_model`)
- **"are"** → `[0.4, 0.2, 0.8, ...]`  (Size: `d_model`)
- **"you"** → `[0.9, 0.1, 0.3, ...]`  (Size: `d_model`)

At this point, each word is just an **embedding vector of size `d_model`**.

> **Note:** Positional encoding is **added** to each embedding to retain word order information.

---

## **Step 2: Creating Query (Q), Key (K), and Value (V)**
For **each word**, we create three separate vectors:

- **Query (Q)** → Determines how much focus a word should get.
- **Key (K)** → Helps decide how much attention a word receives from other words.
- **Value (V)** → Contains the actual word information to be passed on.

These vectors are computed using linear transformations:

Q = W_q * X


K = W_k * X


V = W_v * X

where \( W_q, W_k, W_v \) are **learnable weight matrices**. Whereas, X represents the input embeddings of the words/tokens in the sentence.

For example, for the word **"Hi"**, we get:
- Query vector **Q_hi** (size: `d_model` = 512D)
- Key vector **K_hi** (size: `d_model` = 512D)
- Value vector **V_hi** (size: `d_model` = 512D)

This same process applies to all other words.

---

## **Step 3: Compute Attention Scores in One Head**
Now, we compare how much each word should **attend to** every other word in the sentence. This is done by computing the **dot product** between the **query of one word** and the **keys of all words**.

Since **multi-head attention** is used, each head works with a smaller subspace:
- `d_k = d_model / num_heads`
- If `d_model = 512` and `num_heads = 8`, then `d_k = 512 / 8 = 64`.
- Each head gets **Q, K, V vectors of size 64D**.

| Word  | Query (Q)  | Key (K)  | Value (V)  |
|-------|-----------|----------|------------|
| Hi    | Q_hi (64D) | K_hi (64D) | V_hi (64D) |
| How   | Q_how (64D) | K_how (64D) | V_how (64D) |
| Are   | Q_are (64D) | K_are (64D) | V_are (64D) |
| You   | Q_you (64D) | K_you (64D) | V_you (64D) |

### **Step 3.1: Compute Raw Scores**
For word **Hi**, we compute the dot product of its query with the keys of all words:

Score1= Q_hi.K_hi

Score2= Q_hi.K_how

Score3= Q_hi.K_are

Score4= Q_hi.K_you

### **Step 3.2: Apply Softmax**
The scores are **scaled** to avoid large gradients by dividing by \( \sqrt{d_k} \) and then passed through **softmax** to get probabilities:

$$
\text{Attention Weight} = \text{softmax} \left( \frac{Q \cdot K^T}{\sqrt{d_k}} \right)
$$

$$
\text{Attention Weight}_i = \text{softmax} \left(Score_i\right)
$$

Each word now has an **attention score** that tells how much focus it should give to the other words.

### **Step 3.3: Compute Final Weighted Sum**
Multiply each attention weight by the corresponding **Value (V) vector**:

Output1= Weight1*V_hi

Output2= Weight2*V_how

Output3= Weight2*V_are

Output4= Weight4*V_you

Final embedding for the word **Hi** is: Output1 + Output2 + Output3 + Output4 

Each output is **64D**, so we get **one 64D vector per word per attention head**.

> **Note:** We have Calculated Final Contexual Embeddings of word Hi. Similarly, we have to for other words how, are, you. This is done by computing the **dot product** between the **query of one word** and the **keys of all words**.
---

## **Step 4: Multi-Head Attention**
Since we have **8 heads**, each computes **separate self-attention** and gives an output of size **(batch, seq_length, d_k) = (1, 4, 64)**.

After processing all heads, their outputs are **concatenated** to restore the original `d_model = 512`: **(1, 4, 512)**

The final **multi-head attention output** has the same size as the input embeddings (`d_model = 512`).


