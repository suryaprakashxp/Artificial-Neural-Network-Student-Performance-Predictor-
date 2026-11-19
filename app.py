"""Student Performance Predictor - Activity 2 (Artificial Neural Network)
A web-based interactive demo built with Streamlit + TensorFlow/Keras.
Design a network, train it on synthetic student records, then predict
whether a student will PASS or FAIL and see which habits matter most."""

import numpy as np
import streamlit as st
import tensorflow as tf
import matplotlib.pyplot as plt

st.set_page_config(page_title="Student Performance Predictor", page_icon="🎓", layout="wide")

FEATURES = ["Study hours/day", "Attendance %", "Previous score", "Sleep hours/day"]
ACTIVATIONS = ["relu", "tanh", "sigmoid"]
LEARNING_RATES = [0.001, 0.01, 0.03, 0.1, 0.5]


# ---------------------------------------------------------------- data
def generate_data(n, flip, seed):
    """Synthetic students. Pass probability rises with study, attendance and
    previous score, and peaks near 7 h of sleep (too little or too much hurts)."""
    rng = np.random.default_rng(seed)
    study = rng.uniform(0, 10, n)
    attend = rng.uniform(40, 100, n)
    prev = rng.uniform(30, 100, n)
    sleep = rng.uniform(4, 10, n)
    z = (0.55 * (study - 5) + 0.05 * (attend - 75) + 0.06 * (prev - 60)
         - 0.2 * (sleep - 7) ** 2 + 0.8)
    y = (rng.random(n) < 1 / (1 + np.exp(-z))).astype(int)
    flips = rng.random(n) < flip            # label noise: unpredictable outcomes
    y = np.where(flips, 1 - y, y)
    X = np.c_[study, attend, prev, sleep]
    idx = rng.permutation(n)
    X, y = X[idx].astype("float32"), y[idx].astype("float32")
    split = int(0.8 * n)                    # 80% train / 20% test
    return X[:split], y[:split], X[split:], y[split:]


# --------------------------------------------------------------- model
def build_model(hidden_layers, neurons, activation, lr, dropout):
    tf.keras.utils.set_random_seed(1)
    model = tf.keras.Sequential([tf.keras.layers.Input(shape=(4,))])
    for _ in range(hidden_layers):
        model.add(tf.keras.layers.Dense(neurons, activation=activation))
        if dropout > 0:
            model.add(tf.keras.layers.Dropout(dropout))
    model.add(tf.keras.layers.Dense(1, activation="sigmoid"))
    model.compile(optimizer=tf.keras.optimizers.Adam(lr),
                  loss="binary_crossentropy", metrics=["accuracy"])
    return model


def new_run(cfg):
    Xtr, ytr, Xte, yte = generate_data(cfg["samples"], cfg["flip"], 7)
    mu, sd = Xtr.mean(axis=0), Xtr.std(axis=0)      # standardise using train stats
    st.session_state.update(
        cfg=cfg, mu=mu, sd=sd,
        Xtr=(Xtr - mu) / sd, ytr=ytr, Xte=(Xte - mu) / sd, yte=yte,
        model=build_model(cfg["layers"], cfg["neurons"], cfg["activation"],
                          cfg["lr"], cfg["dropout"]),
        epoch=0, train_loss=[], test_loss=[], train_acc=[], test_acc=[],
    )


def train(epochs):
    s = st.session_state
    for _ in range(epochs):
        h = s.model.fit(s.Xtr, s.ytr, epochs=1, batch_size=16, shuffle=True, verbose=0,
                        validation_data=(s.Xte, s.yte))
        s.epoch += 1
        s.train_loss.append(h.history["loss"][0])
        s.test_loss.append(h.history["val_loss"][0])
        s.train_acc.append(h.history["accuracy"][0])
        s.test_acc.append(h.history["val_accuracy"][0])


# ------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("⚙️ Network settings")
    cfg = dict(
        samples=st.slider("Number of students", 200, 1000, 500, 100),
        flip=st.slider("Random outcomes (label noise)", 0.0, 0.2, 0.05, 0.01),
        layers=st.slider("Hidden layers", 1, 3, 2),
        neurons=st.slider("Neurons per layer", 1, 32, 8),
        activation=st.selectbox("Activation function", ACTIVATIONS),
        lr=st.select_slider("Learning rate", LEARNING_RATES, value=0.01),
        dropout=st.slider("Dropout", 0.0, 0.5, 0.0, 0.1),
    )
    st.caption("Changing any setting builds a fresh, untrained network.")
    epochs_per_click = st.slider("Epochs per 'Train' click", 1, 100, 20)
    c1, c2 = st.columns(2)
    do_train = c1.button("▶ Train", use_container_width=True, type="primary")
    do_step = c2.button("⏭ 1 epoch", use_container_width=True)
    do_reset = st.button("🔄 Reset network", use_container_width=True)

if "cfg" not in st.session_state or st.session_state.cfg != cfg or do_reset:
    new_run(cfg)

if do_train:
    with st.spinner("Training..."):
        train(epochs_per_click)
elif do_step:
    train(1)

s = st.session_state
model = s.model

# ---------------------------------------------------------------- main
st.title("🎓 Student Performance Predictor")
st.caption("Train a neural network on synthetic student records to predict PASS / FAIL, "
           "then try your own what-if scenarios.")

m1, m2, m3, m4 = st.columns(4)
m1.metric("Epoch", s.epoch)
m2.metric("Train loss", f"{s.train_loss[-1]:.4f}" if s.train_loss else "–")
m3.metric("Train accuracy", f"{s.train_acc[-1]*100:.1f}%" if s.train_acc else "–")
m4.metric("Test accuracy", f"{s.test_acc[-1]*100:.1f}%" if s.test_acc else "–")

left, right = st.columns(2)

with left:
    st.subheader("Learning curves")
    if s.epoch:
        fig, (a1, a2) = plt.subplots(2, 1, figsize=(6, 5.5), sharex=True)
        a1.plot(s.train_loss, label="train"); a1.plot(s.test_loss, label="test")
        a1.set_ylabel("Loss"); a1.legend(); a1.grid(alpha=.3)
        a2.plot(s.train_acc, label="train"); a2.plot(s.test_acc, label="test")
        a2.set_ylabel("Accuracy"); a2.set_xlabel("Epoch"); a2.set_ylim(0, 1.02)
        a2.legend(); a2.grid(alpha=.3)
        st.pyplot(fig); plt.close(fig)
    else:
        st.info("Press **Train** in the sidebar to start learning.")

    st.subheader("What-if prediction")
    w1, w2 = st.columns(2)
    study = w1.slider(FEATURES[0], 0.0, 10.0, 5.0, 0.5)
    attend = w2.slider(FEATURES[1], 40, 100, 75)
    prev = w1.slider(FEATURES[2], 30, 100, 60)
    sleep = w2.slider(FEATURES[3], 4.0, 10.0, 7.0, 0.5)
    x = (np.array([[study, attend, prev, sleep]], dtype="float32") - s.mu) / s.sd
    p = float(model(x, training=False).numpy()[0, 0])
    st.progress(p, text=f"Probability of passing: {p*100:.1f}%")
    if s.epoch == 0:
        st.caption("The network is untrained, so this is just a random guess.")
    elif p >= 0.5:
        st.success("Prediction: PASS ✅")
    else:
        st.error("Prediction: FAIL ❌")

with right:
    st.subheader("Test-set confusion matrix")
    pred = (model(s.Xte, training=False).numpy().ravel() >= 0.5).astype(int)
    yt = s.yte.astype(int)
    cm = np.array([[((yt == i) & (pred == j)).sum() for j in (0, 1)] for i in (0, 1)])
    fig, ax = plt.subplots(figsize=(4.2, 3.6))
    ax.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=14,
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["FAIL", "PASS"])
    ax.set_yticks([0, 1]); ax.set_yticklabels(["FAIL", "PASS"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    st.pyplot(fig); plt.close(fig)

    st.subheader("Which factors matter most?")
    if s.epoch:
        base = ((model(s.Xte, training=False).numpy().ravel() >= 0.5) == s.yte).mean()
        rng = np.random.default_rng(0)
        drops = []
        for k in range(4):                   # permutation importance
            d = []
            for _ in range(5):
                Xp = s.Xte.copy()
                Xp[:, k] = rng.permutation(Xp[:, k])
                acc = ((model(Xp, training=False).numpy().ravel() >= 0.5) == s.yte).mean()
                d.append(base - acc)
            drops.append(np.mean(d))
        fig, ax = plt.subplots(figsize=(5.5, 3))
        ax.barh(FEATURES, drops, color="#2563eb")
        ax.set_xlabel("Drop in test accuracy when shuffled"); ax.grid(alpha=.3, axis="x")
        st.pyplot(fig); plt.close(fig)
    else:
        st.info("Train the network to see feature importance.")

    st.subheader("Model summary")
    st.write(f"**{s.cfg['layers']}** hidden layer(s) × **{s.cfg['neurons']}** neurons, "
             f"**{s.cfg['activation']}** activation, dropout {s.cfg['dropout']}, sigmoid output, "
             f"Adam optimizer (lr = {s.cfg['lr']}). Trainable parameters: **{model.count_params()}**.")

with st.expander("💡 What to try"):
    st.markdown("""
- **1 neuron vs 8 neurons**: sleep has a curved effect, so a tiny network misses it.
- Train 100+ epochs with **no dropout**, then add dropout 0.3 and compare train vs test accuracy.
- Raise **random outcomes** to 0.2: accuracy can't reach 100% because some results are unpredictable.
- Learning rate **0.5** makes the loss jump around; **0.001** learns very slowly.
- Use the what-if sliders: set sleep to 4 h vs 7 h and watch the pass probability change.
""")
