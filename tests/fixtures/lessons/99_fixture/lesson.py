# %% [markdown]
# # Fixture lesson
#
# Inline math $x^2$ and display math:
#
# $$\sum_{i=1}^{n} i = \frac{n(n+1)}{2}$$

# %% tags=["include:setup"]

# %%
answer = 21 * 2

# %% tags=["only-kaggle"]
platform = "kaggle"

# %% tags=["only-colab"]
platform = "colab"

# %% tags=["solution"]
assert answer == 42
