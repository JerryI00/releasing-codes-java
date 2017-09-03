/**
 * AdaptiveModule.java
 * 
 * This class contains some adaptive related functions
 */

package jmetal.metaheuristics.moead;

import jmetal.core.SolutionSet;

public class AdaptiveModule {
	
	/**
	 * main function of the proposed AOS system, i.e., Fitness-Rate-Rank-based Multi-Armed Bandit (FRRMAB)
	 * @param quality
	 * @param rewards
	 * @param strategy_usage
	 * @param numStrategies_
	 * @param scale_
	 * @return
	 */
	static int FRRMAB(double[] quality, double[] rewards, int[] strategy_usage,
			int numStrategies_, double scale_) {
		int i;
		double temp1, temp2, temp3;
		int total_usage;
		int best_index;

		total_usage = 0;
		for (i = 0; i < numStrategies_; i++) {
			total_usage += strategy_usage[i];
		}

		for (i = 0; i < numStrategies_; i++) {
			temp1 = 2 * Math.log(total_usage);
			temp2 = temp1 / strategy_usage[i];
			temp3 = Math.sqrt(temp2);
			quality[i] = rewards[i] + scale_ * temp3;
		}

		best_index = 0;
		for (i = 1; i < numStrategies_; i++) {
			if (quality[i] > quality[best_index]) {
				best_index = i;
			}
		}
		best_index++;

		return best_index;
	}

	/**
	 * Update the reward values, read from the sliding window
	 * @param slidingWindow_
	 * @param reward_
	 * @param strategyUsgae_
	 * @param curSize_
	 */
	static void update_rewards(double[][] slidingWindow_, double[] reward_,
			int[] strategyUsgae_, int curSize_) {
		int i;
		int index;
		double fitnessImprovement;

		for (i = 0; i < curSize_; i++) {
			if (slidingWindow_[0][i] == -1 || slidingWindow_[1][i] == -1) {
				System.out.println("!!!!!!");
			}
			index = (int) slidingWindow_[0][i];
			fitnessImprovement = slidingWindow_[1][i];
			switch (index) {
			case 1:
				reward_[0] += fitnessImprovement;
				strategyUsgae_[0]++;
				break;
			case 2:
				reward_[1] += fitnessImprovement;
				strategyUsgae_[1]++;
				break;
			case 3:
				reward_[2] += fitnessImprovement;
				strategyUsgae_[2]++;
				break;
			case 4:
				reward_[3] += fitnessImprovement;
				strategyUsgae_[3]++;
				break;
			}
		}
	}

	/**
	 * Rank the reward values
	 * @param reward_
	 * @param numStrategies_
	 * @param rank
	 */
	static void Rank_rewards(double[] reward_, int numStrategies_, int[] rank) {
		int i, j;
		double[][] temp;
		double temp_index;
		double temp_value;

		temp = new double[2][numStrategies_];
		for (i = 0; i < numStrategies_; i++) {
			temp[0][i] = reward_[i];
			temp[1][i] = i;
		}

		for (i = 0; i < numStrategies_ - 1; i++) {
			for (j = i + 1; j < numStrategies_; j++) {
				if (temp[0][i] < temp[0][j]) {
					temp_value = temp[0][j];
					temp[0][j] = temp[0][i];
					temp[0][i] = temp_value;

					temp_index = temp[1][j];
					temp[1][j] = temp[1][i];
					temp[1][i] = temp_index;
				}
			}
		}

		for (i = 0; i < numStrategies_; i++) {
			rank[i] = (int) temp[1][i];
		}
	}

	static void CreaditAssignmentDecay(double[] strategy_rewards,
			double[] decay_rewards, int numStrategies_, int[] rank,
			double decayFactor) {
		int i;
		double decayed, decay_sum;
		double[] decay_value;

		decay_value = new double[numStrategies_];

		for (i = 0; i < numStrategies_; i++) {
			decayed = Math.pow(decayFactor, i);
			switch (rank[i]) {
			case 0:
				decay_value[0] = strategy_rewards[0] * decayed;
				break;
			case 1:
				decay_value[1] = strategy_rewards[1] * decayed;
				break;
			case 2:
				decay_value[2] = strategy_rewards[2] * decayed;
				break;
			case 3:
				decay_value[3] = strategy_rewards[3] * decayed;
				break;
			}
		}

		decay_sum = 0.0;
		for (i = 0; i < numStrategies_; i++) {
			decay_sum += decay_value[i];
		}

		for (i = 0; i < numStrategies_; i++) {
			if (decay_sum == 0) {
				decay_rewards[i] = 0.0;
			} else {
				decay_rewards[i] = decay_value[i] / decay_sum;
			}
		}
	}
}
