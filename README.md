# POLICY ITERATION ALGORITHM

## AIM
The goal of the notebook is to implement and evaluate a policy iteration algorithm within a custom environment (gym-walk) to find the optimal policy that maximizes the agent's performance in terms of reaching a goal state with the highest probability and reward.

## PROBLEM STATEMENT
The task is to develop and apply a policy iteration algorithm to solve a grid-based environment (gym-walk). The environment consists of states the agent must navigate through to reach a goal. The agent has to learn the best sequence of actions (policy) that maximizes its chances of reaching the goal state while obtaining the highest cumulative reward.

## POLICY ITERATION ALGORITHM
Initialize: Start with a random policy for each state and initialize the value function arbitrarily.

Policy Evaluation: For each state, evaluate the current policy by computing the expected value function under the current policy.

Policy Improvement: Improve the policy by making it greedy with respect to the current value function (i.e., choose the action that maximizes the value function for each state).

Check Convergence: Repeat the evaluation and improvement steps until the policy stabilizes (i.e., when no further changes to the policy occur).

Optimal Policy: Once convergence is achieved, the policy is considered optimal, providing the best actions for the agent in each state.


## POLICY IMPROVEMENT FUNCTION
### Name : SABARI S
### Register Number : 212222240085
```python
Include the policy improvement function
def policy_improvement(V, P, gamma=1.0):
    Q = np.zeros((len(P), len(P[0])), dtype=np.float64)
    for s in range(len(P)):
        for a in range(len(P[s])):
            for prob, next_state, reward, done in P[s][a]:
                Q[s, a] += prob * (reward + gamma * V[next_state] * (not done))

    new_pi = lambda s: np.argmax(Q[s, :])
    return new_pi

```
## POLICY ITERATION FUNCTION
### Name : SABARI S
### Register Number : 212222240085
```python
Include the policy iteration function
def policy_iteration(P, gamma=1.0, theta=1e-10):
    num_states = len(P)
    num_actions = len(P[0])

    # Initialize an arbitrary policy (e.g., all actions are 0 - LEFT)
    pi = lambda s: 0

    while True:
        # Policy Evaluation
        V = policy_evaluation(pi, P, gamma, theta)

        # Policy Improvement
        new_pi_func = policy_improvement(V, P, gamma)

        # Check for policy convergence
        policy_stable = True
        for s in range(num_states):
            if new_pi_func(s) != pi(s):
                policy_stable = False
                break

        pi = new_pi_func

        if policy_stable:
            break

    return V, pi

```

## OUTPUT:
### 1. Policy, Value function and success rate for the Adversarial Policy
## Policy
<img width="503" height="161" alt="image" src="https://github.com/user-attachments/assets/778e53f3-a9ca-4cde-90d1-c11d094d0ef2" />
## success rate
<img width="868" height="138" alt="image" src="https://github.com/user-attachments/assets/80536da5-62b8-46ad-8585-b0944b8a1f95" />
## Value function
<img width="572" height="168" alt="image" src="https://github.com/user-attachments/assets/bd077728-8539-4fc2-b8c8-8699cfe97d7d" />




### 2. Policy, Value function and success rate for the Improved Policy
## Policy
<img width="531" height="162" alt="image" src="https://github.com/user-attachments/assets/aeb9496c-5816-4f55-9165-43144ad82cae" />

## success rate
<img width="886" height="138" alt="image" src="https://github.com/user-attachments/assets/e5c76ad7-8a24-4b88-8eb9-e3f07a87ff9d" />

## Value function
<img width="566" height="175" alt="image" src="https://github.com/user-attachments/assets/ece622d9-505d-4eb9-bf79-f332b0f349b6" />





### 3. Policy, Value function and success rate after policy iteration
## Policy
<img width="536" height="185" alt="image" src="https://github.com/user-attachments/assets/9c2120e7-be94-4954-b5ed-b31212778361" />


## success rate
<img width="945" height="120" alt="image" src="https://github.com/user-attachments/assets/8e20950b-d8ef-45d6-9507-11e8df332483" />


## Value function
<img width="536" height="161" alt="image" src="https://github.com/user-attachments/assets/b770c01c-958a-49bf-90c8-be096a70e6ab" />




## RESULT:

Thus the program to iterate the policy evaluation and policy improvement is executed successfully.
