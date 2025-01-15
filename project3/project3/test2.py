class ReinforceBaselineAgent(ReinforceAgent):

    ##############################################################################
    #                             ENTER YOUR CODE                               #
    ##############################################################################
    def __init__(self, n_actions, learning_rate=0.01, gamma=0.99):
        # Initialize policy parameters (theta) as a vector of zeros
        self.theta = np.zeros(n_actions)
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.n_actions = n_actions

    def policy(self, state):
        """
        Define a softmax policy over actions based on current state.
        """
        # Use a simple linear model with theta
        logits = self.theta
        return softmax(logits)

    def select_action(self, state):
        """
        Select an action based on the policy.
        """
        probs = self.policy(state)
        return np.random.choice(self.n_actions, p=probs)

    def compute_returns(self, rewards):
        """
        Compute the discounted returns (G_t) for each time step.
        """
        returns = np.zeros_like(rewards)
        G = 0
        for t in reversed(range(len(rewards))):
            G = rewards[t] + self.gamma * G
            returns[t] = G
        return returns

    def update_policy(self, states, actions, returns):
        """
        Update the policy using the REINFORCE algorithm.
        """
        for t in range(len(states)):
            state = states[t]
            action = actions[t]
            G = returns[t]
            probs = self.policy(state)
            prob_action = probs[action]
            # Compute the gradient and update theta
            gradient = -np.log(prob_action) * G
            self.theta += self.learning_rate * gradient

    def train(self, env, num_episodes=1000):
        """
        Train the agent using REINFORCE algorithm.
        """
        rewards_list = []

        for episode in tqdm(range(num_episodes)):
            states = []
            actions = []
            rewards = []

            env.reset()
            done = False
            while not done:
                state = env.state
                action = self.select_action(state)
                states.append(state)
                actions.append(action)

                reward, done = env.step(action == 1)  # Action 1 means "go right"
                rewards.append(reward)

            returns = self.compute_returns(rewards)
            self.update_policy(states, actions, returns)

            total_reward = np.sum(rewards)
            rewards_list.append(total_reward)

        return rewards_list


# Running the environment and training the agent
env = ShortCorridor()
agent = ReinforceAgent(n_actions=2)

# Train agent
rewards = agent.train(env)

# Plotting the total rewards over episodes
plt.plot(rewards)
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.title('Training Performance')
plt.savefig('reinforce_training_performance.png')


