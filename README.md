# Policy based RL
In this project we.
- Implemented and compare the performance of SARSA and expected SARSA on the Frozen Lake domain from the Gym environment suite
- Implemented and compare empirically Q-learning and actor-critic with linear function approximation on the cart-pole domain from the Gym environment suite

## Tabular RL 
Tabular Reinforcement Learning (RL) refers to a class of reinforcement learning methods that use tables to represent the value functions or policies of an agent. In this approach, the agent maintains a table, often called a Q-table or state-value table, where each entry corresponds to an estimate of the value of a particular state or state-action pair. Tabular RL is particularly useful when the state and action spaces are small and discrete, making it possible to represent and update the values for all state-action pairs efficiently. 

### Environment: OpenAI Frozen Lake
The "FrozenLake" environment is one of the classic environments provided by the OpenAI Gym toolkit. It is a grid-world environment in which an agent navigates on a frozen lake, trying to reach the goal while avoiding holes. The environment is often used as a benchmark for testing reinforcement learning algorithms. 

### Algorithm

#### SARSA
SARSA is a reinforcement learning algorithm used for solving Markov Decision Processes (MDPs) and sequential decision-making problems. The name "SARSA" stands for "State-Action-Reward-State-Action," which indicates how the algorithm updates its Q-values based on the observed experiences. 

In SARSA, the agent learns an action-value function (Q-function), which estimates the expected cumulative reward that can be obtained by starting from a specific state, taking a specific action, and then following a specific policy. The algorithm follows an on-policy learning approach, which means that it learns the Q-values based on its own policy's actions. 

Here's a step-by-step explanation of the SARSA algorithm: 

1. **Initialization:** Initialize the Q-values for all state-action pairs arbitrarily or to some specific initial values. 

2. **Choose an Action:** Observe the current state, s. 

3. **Select an Action:** Based on the current state, s, and using a policy (often epsilon-greedy or softmax), select an action, a, to execute in the environment. 

4. **Execute the Action:** Take action, a, and observe the next state, s', and the reward, r. 

5. **Select Next Action:** Based on the next state, s', and using the same policy, select the next action, a', to execute in the environment. 

6. **Update Q-value:** Update the Q-value of the current state-action pair using the SARSA update rule:
   - Q(s, a) ← Q(s, a) + α * [r + γ * Q(s', a') - Q(s, a)] where:
   - α (alpha) is the learning rate that controls the step size of the update.
   - γ (gamma) is the discount factor that balances immediate and future rewards. 

8. **Set Current State and Action:** Set the current state, s, and action, a, to the next state, s', and action, a'. 

9. **Repeat:** Continue steps 3 to 7 until the agent reaches a terminal state or a certain stopping condition is met. 

The SARSA algorithm is an incremental learning algorithm, meaning it gradually updates the Q-values based on the agent's interactions with the environment. By continually exploring the environment and updating the Q-values using the observed experiences, the agent learns to improve its policy over time, ultimately converging to an optimal policy. 

#### Expected SARSA
Expected SARSA is another reinforcement learning algorithm that is closely related to both SARSA and Q-learning. Like SARSA, it is an on-policy algorithm, meaning it learns the Q-values based on the current policy. The key difference between Expected SARSA and SARSA lies in how the next action is chosen during the value update step. 

In SARSA, the next action is selected using the same policy that was used to select the current action. However, in Expected SARSA, the next action is chosen by considering the expected value of all possible actions in the next state, weighted by their probabilities under the current policy. 

The Expected SARSA update rule is as follows: 

Q(s, a) ← Q(s, a) + α * [r + γ * Σ[π(a'|s') * Q(s', a')] - Q(s, a)] 

where: 
- α (alpha) is the learning rate. 
- γ (gamma) is the discount factor. 
- Σ[π(a'|s') * Q(s', a')] represents the expected value of the Q-function in the next state, s', considering all possible actions a' weighted by their probabilities according to the current policy. 

In other words, rather than using the Q-value of the specific action taken in the next state (as in SARSA), Expected SARSA considers the expected value of all actions in the next state and updates the Q-value accordingly. 

The advantage of Expected SARSA over SARSA is that it can reduce variance in the learning process since it takes the expected value of the Q-function over all possible actions in the next state. This can lead to more stable learning and potentially better convergence in certain environments, especially when the policy is stochastic and exploration is an essential aspect of the learning process. 

 
## Function approximation 

Function approximation in Reinforcement Learning (RL) is a technique used to represent value functions or policies in a more compact and generalizable way, especially when dealing with large or continuous state and action spaces. Instead of using tabular representations, which require storing values for each individual state or state-action pair, function approximation approximates the value functions using a parameterized function. 

In function approximation, the goal is to learn a function (e.g., a neural network) that takes the state or state-action features as inputs and outputs the corresponding value function estimates. The function is usually parameterized by a set of weights, and the learning process involves updating these weights based on the agent's interactions with the environment. 

 
### Environment: OpenAI Frozen Lake Cart-Pole
The "CartPole" environment is another classic reinforcement learning environment provided by the OpenAI Gym toolkit. It simulates a simple physics problem where an inverted pendulum (pole) is attached to a moving cart. The goal of the agent is to balance the pole on the cart by applying appropriate forces to the cart. 

### Algorithm
#### Q-learning
Q-learning is a popular and widely used reinforcement learning algorithm that enables an agent to learn an optimal action-value function (Q-function) for decision-making in Markov Decision Processes (MDPs). The Q-function estimates the expected cumulative reward an agent can achieve by taking a specific action in a given state and then following the optimal policy from that point onward. 

The Q-learning algorithm is model-free, meaning it does not require knowledge of the underlying dynamics of the environment and learns directly from interactions with the environment. It belongs to the class of off-policy learning algorithms, which means that it learns the optimal Q-values independently of the policy being followed during exploration. 

Here's a step-by-step explanation of the Q-learning algorithm: 

1. **Initialization:** Initialize the Q-values for all state-action pairs arbitrarily or to some specific initial values. 

2. **Choose an Action:** Observe the current state, s. 

3. **Select an Action:** Based on the current state, s, and using an exploration strategy (commonly epsilon-greedy), select an action, a, to execute in the environment. 

4. **Execute the Action:** Take action, a, and observe the next state, s', and the reward, r. 

5. **Update Q-values:** Update the Q-value of the current state-action pair using the Q-learning update rule: Q(s, a) ← Q(s, a) + α * [r + γ * max(Q(s', a')) - Q(s, a)] where:
   - α (alpha) is the learning rate that controls the step size of the update.
   - γ (gamma) is the discount factor that balances immediate and future rewards.
   - max(Q(s', a')) represents the maximum Q-value of all possible actions in the next state, s'. 

6. **Set Current State:** Set the current state, s, to the next state, s'.
   
7. **Repeat:** Continue steps 3 to 6 until the agent reaches a terminal state or a certain stopping criterion is met. 

Q-learning iteratively updates the Q-values as the agent interacts with the environment, allowing it to learn an optimal Q-function. By following a suitable exploration strategy, Q-learning ensures a good balance between exploration and exploitation, enabling the agent to gradually improve its policy over time. 

#### Actor-Critic with linear function approximation
Actor-Critic with linear function approximation is a reinforcement learning algorithm that combines the advantages of both policy-based methods (actor) and value-based methods (critic) while using a linear function approximation to represent the policy and value functions. This approach is useful in scenarios with large state and action spaces where tabular methods are infeasible. 

Here's how the Actor-Critic algorithm with linear function approximation works: 

- **Actor:** The actor is responsible for learning and improving the policy. Instead of directly representing the policy as a table, it uses a linear function approximation to estimate the policy parameters, denoted as θ. The policy function can be written as π(a|s,θ), which estimates the probability of taking action 'a' in state 's' using the current policy parameters θ.
- **Critic:** The critic is responsible for estimating the value function V(s). It also uses linear function approximation, where the value function is represented as V(s) ≈ w^T * φ(s), where 'w' is a weight vector, and φ(s) is the feature vector representation of state 's'. 

1. **Initialization:** Initialize the policy parameters θ and value function weights 'w' to some initial values. 

2. **Interaction with the Environment:** The agent interacts with the environment by executing actions based on the current policy and observing the resulting rewards and next states. 

3. **Update the Actor (Policy):** The actor is updated using the policy gradient method, which aims to maximize the expected cumulative reward. The update rule is:
   - θ ← θ + α * ∇θ log(π(a|s,θ)) * A(s,a) where:
   - α (alpha) is the learning rate.
   - A(s, a) is the advantage function, which represents how much better or worse the action 'a' is compared to the average action in state 's'. It is typically estimated as the difference between the observed return (total reward) and the value function estimate for the state-action pair. 

4. **Update the Critic (Value Function):** The critic is updated using the temporal difference (TD) learning method. The update rule is: w ← w + β * [R + γ * V(s') - V(s)] * φ(s) where:
   - β is the learning rate for the critic.
   - R is the observed reward after taking action 'a' in state 's'.
   - γ is the discount factor for future rewards.
   - V(s') is the estimated value of the next state. 

5. **Repeat:** Continue steps 4 to 6 until the agent reaches a terminal state or a certain stopping criterion is met. 

By combining policy-based and value-based methods, the Actor-Critic algorithm with linear function approximation can be more stable and efficient in learning optimal policies compared to using each method individually. It is commonly used in practical RL applications and serves as a foundation for more complex and powerful deep reinforcement learning algorithms. 

 
