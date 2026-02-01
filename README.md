## Code Structure

The codebase is organized into three main components:

- **SimCLR baseline**  
  Standard SimCLR training for self-supervised speech representation learning.

- **SimCLR + adversarial debiasing**  
  Extends the baseline with adversarial classifiers to remove demographic information
  (e.g., gender, age, accent) from the learned embeddings.

- **SimCLR + adversarial debiasing + causal bottleneck**  
  Further augments the adversarial setup with a causal bottleneck layer to explicitly
  separate demographic and task-relevant representations.

Each component is implemented in a **separate directory**.

