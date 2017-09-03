/**
 * ssNSGAII_ENLU.java
 *
 * This is main implementation of a steady-state NSGA-II where the non-dominated sorting
 * procedure is replaced by the efficient non-domination level update procedure (ENLU).
 *
 * Author:
 * 		Ke Li <k.li@exeter.ac.uk>
 *
 * Affliation:
 * 		Department of Computer Science, University of Exeter
 *
 * Reference:
 * 		K. Li, K. Deb, Q. Zhang, Q. Zhang,
 * 		"Efficient Non-domination Level Update Method for Steady-State Evolutionary Multiobjective Optimization"
 * 		IEEE Transactions on Cybernetics (TCYB), 47(9): 2838-2849, 2017.
 *
 * Homepage:
 * 		https://coda-group.github.io/
 *
 * Copyright (c) 2017 Ke Li
 *
 * Note: This is a free software developed based on the open source project
 * jMetal<http://jmetal.sourceforge.net>. The copy right of jMetal belongs to
 * its original authors, Antonio J. Nebro and Juan J. Durillo. Nevertheless,
 * this current version can be redistributed and/or modified under the terms of
 * the GNU Lesser General Public License as published by the Free Software
 * Foundation, either version 3 of the License, or (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 *
 */

package jmetal.metaheuristics.nsgaII;

import java.util.Vector;

import jmetal.core.*;
import jmetal.util.comparators.CrowdingComparator;
import jmetal.util.*;

/**
 * This class implements a steady-state version of NSGA-II.
 */
public class ssNSGAII_ENLU extends Algorithm {

	private int populationSize_;

	private SolutionSet union_;
	private SolutionSet population_;

	int numRanks;

	int[][] rankIdx_;	// index matrix for the non-domination levels

	/**
	 * Constructor
	 *
	 * @param problem: Problem to solve
	 */
	public ssNSGAII_ENLU(Problem problem) {
		super(problem);
	} // NSGAII

	/**
	 * Execution of the steady-state NSGA-II.
	 *
	 * @return a <code>SolutionSet</code> that is a set of non dominated
	 *         solutions as a result of the algorithm execution
	 * @throws JMException
	 */
	public SolutionSet execute() throws JMException, ClassNotFoundException {
		int maxEvaluations;
		int evaluations;

		int requiredEvaluations; // Use in the example of use of the
		// indicators object (see below)

		SolutionSet tempPop;
		SolutionSet offspringPopulation;

		Operator mutationOperator;
		Operator crossoverOperator;
		Operator selectionOperator;

		Distance distance = new Distance();

		// Read the parameters
		populationSize_ = ((Integer) getInputParameter("populationSize")).intValue();
		maxEvaluations  = ((Integer) getInputParameter("maxEvaluations")).intValue();

		// Initialize the variables
		population_  = new SolutionSet(populationSize_);
		tempPop      = new SolutionSet(populationSize_);
		evaluations  = 0;

		rankIdx_	 = new int[populationSize_][populationSize_];

		requiredEvaluations = 0;

		// Read the operators
		mutationOperator  = operators_.get("mutation");
		crossoverOperator = operators_.get("crossover");
		selectionOperator = operators_.get("selection");

		// Create the initial solutionSet
		Solution newSolution;
		for (int i = 0; i < populationSize_; i++) {
			newSolution = new Solution(problem_);
			problem_.evaluate(newSolution);
			problem_.evaluateConstraints(newSolution);
			evaluations++;
			population_.add(newSolution);
		} // for

		// Get the non-domination level structure of the initial population
		Ranking ranking = new Ranking(population_);
		int curRank;
		for (int i = 0; i < populationSize_; i++) {
			curRank = population_.get(i).getRank();
			rankIdx_[curRank][i] = 1;
		}

		// Generations ...
		while (evaluations < maxEvaluations) {

			// Create an offSpring
			offspringPopulation = new SolutionSet(populationSize_);
			Solution[] parents 	= new Solution[2];

			// obtain parents
			parents[0] = (Solution) selectionOperator.execute(population_);
			parents[1] = (Solution) selectionOperator.execute(population_);

			// crossover & mutation
			Solution[] offSprings = (Solution[]) crossoverOperator.execute(parents);
			mutationOperator.execute(offSprings[0]);

			// evaluation
			problem_.evaluate(offSprings[0]);
			problem_.evaluateConstraints(offSprings[0]);
			evaluations++;

			// insert child into the offspring population
			offspringPopulation.add(offSprings[0]);

			// update the non-domination level structure
			numRanks = nondominated_sorting_add(offSprings[0]);

			// create the solutionSet 'union' of solutionSet and offSpring
			union_ = ((SolutionSet) population_).union(offspringPopulation);

			// clear rankIdx matrix
			for (int i = 0; i < populationSize_; i++) {
				for (int j = 0; j < populationSize_; j++) {
					rankIdx_[i][j] = 0;
				}
			}

			// Assign crowding distance to solutionSet union
			distance.crowdingDistanceAssignment(union_, problem_.getNumberOfObjectives());

			population_.clear();

			if (numRanks == 1) {	// only one non-domination level
				union_.sort(new CrowdingComparator());
				for (int i = 0; i < populationSize_; i++) {
					population_.add(union_.get(i));
					rankIdx_[union_.get(i).getRank()][i] = 1;
				}
			} else {	// have multiple non-domination levels
				union_.sort(new CrowdingComparator());
				int index = 0;
				for (int i = 0; i < (populationSize_ + 1); i++) {
					if (union_.get(i).getRank() == (numRanks - 1)) {
						tempPop.add(union_.get(i));
					} else {
						population_.add(union_.get(i));
						rankIdx_[union_.get(i).getRank()][index] = 1;
						index++;
					}
				}
				int tempSize = tempPop.size();
				for (int i = 0; i < (tempSize - 1); i++) {
					population_.add(tempPop.get(i));
					rankIdx_[numRanks - 1][index] = 1;
					index++;
				}
			}

			// If the deleted solutions are not the newly generated offspring, we still need to update the non-domination level structure
			if (!union_.get(populationSize_).equals(offSprings[0]))
				nondominated_sorting_delete(union_.get(populationSize_));

			tempPop.clear();
		} // while

		// Return as output parameter the required evaluations
		setOutputParameter("evaluations", requiredEvaluations);

		return population_;
	} // execute

	/**
	 * print the median result
	 * @param idx
	 */
	public int medianPrint(int idx, int evaluations) {
		if (evaluations % 2500 == 0) {
			String str1 = "FUN";
			String str2 = str1 + Integer.toString(idx);

			population_.printObjectivesToFile(str2);
			idx++;
		}

		return idx;
	}

	/**
	 * update the non-domination level when adding a solution
	 *
	 * @param indiv
	 * @return
	 */
	public int nondominated_sorting_add(Solution indiv) {

		int flag = 0;
		int flag1, flag2, flag3;

		// count the number of non-domination levels
		int num_ranks = 0;
		Vector<Integer> frontSize = new Vector<Integer>();
		for (int i = 0; i < populationSize_; i++) {
			int rankCount = countRankOnes(i);
			if (rankCount != 0) {
				frontSize.addElement(rankCount);
				num_ranks++;
			} else {
				break;
			}
		}

		Vector<Integer> dominateList = new Vector<Integer>();	// used to keep the solutions dominated by 'indiv'
		int level = 0;
		for (int i = 0; i < num_ranks; i++) {
			level = i;
			if (flag == 1) {	// 'indiv' is non-dominated with all solutions in the ith non-domination level, then 'indiv' belongs to the ith level
				indiv.setRank(i - 1);
				return num_ranks;
			} else if (flag == 2) {	// 'indiv' dominates some solutions in the ith level, but is non-dominated with some others, then 'indiv' belongs to the ith level, and move the dominated solutions to the next level
				indiv.setRank(i - 1);

				int prevRank = i - 1;

				// process the solutions belong to 'prevRank'th level and are dominated by 'indiv' ==> move them to 'prevRank+1'th level and find the solutions dominated by them
				int curIdx;
				int newRank = prevRank + 1;
				int curListSize = dominateList.size();
				for (int j = 0; j < curListSize; j++) {
					curIdx = dominateList.get(j);
					rankIdx_[prevRank][curIdx] = 0;
					rankIdx_[newRank][curIdx]  = 1;
					population_.get(curIdx).setRank(newRank);
				}
				for (int j = 0; j < populationSize_; j++) {
					if (rankIdx_[newRank][j] == 1) {
						for (int k = 0; k < curListSize; k++) {
							curIdx = dominateList.get(k);
							if (checkDominance(population_.get(curIdx), population_.get(j)) == 1) {
								dominateList.addElement(j);
								break;
							}

						}
					}
				}
				for (int j = 0; j < curListSize; j++)
					dominateList.remove(0);

				// if there are still some other solutions moved to the next level, check their domination situation in their new level
				prevRank 	= newRank;
				newRank  	= newRank + 1;
				curListSize = dominateList.size();
				if (curListSize == 0)
					return num_ranks;
				else {
					int allFlag = 0;
					do {
						for (int j = 0; j < curListSize; j++) {
							curIdx = dominateList.get(j);
							rankIdx_[prevRank][curIdx] = 0;
							rankIdx_[newRank][curIdx]  = 1;
							population_.get(curIdx).setRank(newRank);
						}
						for (int j = 0; j < populationSize_; j++) {
							if (rankIdx_[newRank][j] == 1) {
								for (int k = 0; k < curListSize; k++) {
									curIdx = dominateList.get(k);
									if (checkDominance(population_.get(curIdx), population_.get(j)) == 1) {
										dominateList.addElement(j);
										break;
									}
								}
							}
						}
						for (int j = 0; j < curListSize; j++)
							dominateList.remove(0);

						curListSize = dominateList.size();
						if (curListSize != 0) {
							prevRank = newRank;
							newRank  = newRank + 1;
							if (curListSize == frontSize.get(prevRank)) {	// if all solutions in the 'prevRank'th level are dominated by the newly added solution, move them all to the next level
								allFlag = 1;
								break;
							}
						}
					} while (curListSize != 0);

					if (allFlag == 1) {	// move the solutions after the 'prevRank'th level to their next levels
						int remainSize = num_ranks - prevRank;
						int[][] tempRecord = new int[remainSize][populationSize_];

						int tempIdx = 0;
						for (int j = 0; j < dominateList.size(); j++) {
							tempRecord[0][tempIdx] = dominateList.get(j);
							tempIdx++;
						}

						int k = 1;
						int curRank = prevRank + 1;
						while (curRank < num_ranks) {
							tempIdx = 0;
							for (int j = 0; j < populationSize_; j++) {
								if (rankIdx_[curRank][j] == 1) {
									tempRecord[k][tempIdx] = j;
									tempIdx++;
								}
							}
							curRank++;
							k++;
						}

						k = 0;
						curRank = prevRank;
						while (curRank < num_ranks) {
							int level_size = frontSize.get(curRank);

							int tempRank;
							for (int j = 0; j < level_size; j++) {
								curIdx   = tempRecord[k][j];
								tempRank = population_.get(curIdx).getRank();
								newRank  = tempRank + 1;
								population_.get(curIdx).setRank(newRank);

								rankIdx_[tempRank][curIdx] = 0;
								rankIdx_[newRank][curIdx]  = 1;
							}
							curRank++;
							k++;
						}
						num_ranks++;
					}

					if (newRank == num_ranks)
						num_ranks++;

					return num_ranks;
				}
			} else if (flag == 3 || flag == 0) {	// if 'indiv' is dominated by some solutions in the ith level, skip it, and term to the next level
				flag1 = flag2 = flag3 = 0;
				for (int j = 0; j < populationSize_; j++) {
					if (rankIdx_[i][j] == 1) {
						switch (checkDominance(indiv, population_.get(j))) {
							case 1: {
								flag1 = 1;
								dominateList.addElement(j);
								break;
							}
							case 0: {
								flag2 = 1;
								break;
							}
							case -1: {
								flag3 = 1;
								break;
							}
						}

						if (flag3 == 1) {
							flag = 3;
							break;
						} else if (flag1 == 0 && flag2 == 1)
							flag = 1;
						else if (flag1 == 1 && flag2 == 1)
							flag = 2;
						else if (flag1 == 1 && flag2 == 0)
							flag = 4;
						else
							continue;
					}
				}

			} else {	// (flag == 4) if 'indiv' dominates all solutions in the ith level, solutions in the current level and beyond move their current next levels
				indiv.setRank(i - 1);
				i = i - 1;
				int remainSize = num_ranks - i;
				int[][] tempRecord = new int[remainSize][populationSize_];

				int k = 0;
				while (i < num_ranks) {
					int tempIdx = 0;
					for (int j = 0; j < populationSize_; j++) {
						if (rankIdx_[i][j] == 1) {
							tempRecord[k][tempIdx] = j;
							tempIdx++;
						}
					}
					i++;
					k++;
				}

				k = 0;
				i = indiv.getRank();
				while (i < num_ranks) {
					int level_size = frontSize.get(i);

					int curIdx;
					int curRank, newRank;
					for (int j = 0; j < level_size; j++) {
						curIdx  = tempRecord[k][j];
						curRank = population_.get(curIdx).getRank();
						newRank = curRank + 1;
						population_.get(curIdx).setRank(newRank);

						rankIdx_[curRank][curIdx] = 0;
						rankIdx_[newRank][curIdx] = 1;
					}
					i++;
					k++;
				}
				num_ranks++;

				return num_ranks;
			}
		}
		// if flag is still 3 after the for-loop, it means that 'indiv' is in the current last level
		if (flag == 1) {
			indiv.setRank(level);
		} else if (flag == 2) {
			indiv.setRank(level);

			int curIdx;
			int tempSize = dominateList.size();
			for (int i = 0; i < tempSize; i++) {
				curIdx = dominateList.get(i);
				population_.get(curIdx).setRank(level + 1);

				rankIdx_[level][curIdx] 	= 0;
				rankIdx_[level + 1][curIdx] = 1;
			}
			num_ranks++;
		} else if(flag == 3) {
			indiv.setRank(level + 1);
			num_ranks++;
		} else {
			indiv.setRank(level);
			for (int i = 0; i < populationSize_; i++) {
				if (rankIdx_[level][i] == 1) {
					population_.get(i).setRank(level + 1);

					rankIdx_[level][i] 	   = 0;
					rankIdx_[level + 1][i] = 1;
				}
			}
			num_ranks++;
		}

		return num_ranks;
	}

	/**
	 * update the non-domination level structure after deleting a solution
	 *
	 * @param indiv
	 * @return
	 */
	public void nondominated_sorting_delete(Solution indiv) {

		// find the non-domination level of 'indiv'
		int indivRank = indiv.getRank();

		Vector<Integer> curLevel     = new Vector<Integer>();	// used to keep the solutions in the current non-domination level
		Vector<Integer> dominateList = new Vector<Integer>();	// used to keep the solutions need to be moved

		for (int i = 0; i < populationSize_; i++) {
			if (rankIdx_[indivRank][i] == 1)
				curLevel.addElement(i);
		}

		int flag;
		// find the solutions belonging to the 'indivRank+1'th level and are dominated by 'indiv'
		int investigateRank = indivRank + 1;
		if (investigateRank < numRanks) {
			for (int i = 0; i < populationSize_; i++) {
				if (rankIdx_[investigateRank][i] == 1) {
					flag = 0;
					if (checkDominance(indiv, population_.get(i)) == 1) {
						for (int j = 0; j < curLevel.size(); j++) {
							if (checkDominance(population_.get(i), population_.get(curLevel.get(j))) == -1) {
								flag = 1;
								break;
							}
						}
						if (flag == 0) {	// the ith solution can move to the prior level
							dominateList.addElement(i);
							rankIdx_[investigateRank][i] 	= 0;
							rankIdx_[investigateRank - 1][i] = 1;
							population_.get(i).setRank(investigateRank - 1);
						}
					}
				}
			}
		}

		int curIdx;
		int curListSize = dominateList.size();
		while (curListSize != 0) {
			curLevel.clear();
			for (int i = 0; i < populationSize_; i++) {
				if (rankIdx_[investigateRank][i] == 1)
					curLevel.addElement(i);
			}
			investigateRank = investigateRank + 1;

			if (investigateRank < numRanks) {
				for (int i = 0; i < curListSize; i++) {
					curIdx = dominateList.get(i);
					for (int j = 0; j < populationSize_; j++) {
						if (j == populationSize_)
							System.out.println("Fuck me!!!");
						if (rankIdx_[investigateRank][j] == 1) {
							flag = 0;
							if (checkDominance(population_.get(curIdx), population_.get(j)) == 1) {
								for (int k = 0; k < curLevel.size(); k++) {
									if (checkDominance(population_.get(j), population_.get(curLevel.get(k))) == -1) {
										flag = 1;
										break;
									}
								}
								if (flag == 0) {
									dominateList.addElement(j);
									rankIdx_[investigateRank][j] 	= 0;
									rankIdx_[investigateRank - 1][j] = 1;
									population_.get(j).setRank(investigateRank - 1);
								}
							}
						}
					}
				}
			}
			for (int i = 0; i < curListSize; i++)
				dominateList.remove(0);

			curListSize = dominateList.size();
		}
	}

	/**
	 * Count the # of 1s on a row of rank matrix
	 * @param location
	 * @return
	 */
	public int countRankOnes(int location) {

		int count = 0;
		for (int i = 0; i < populationSize_; i++) {
			if (rankIdx_[location][i] == 1)
				count++;
		}

		return count;
	}

	/**
	 * Check the dominance relationship between 'a' and 'b':
	 * 1 -> 'a' dominates 'b'; 0 -> 'a' and 'b' non-dominated; -1 -> 'b' dominates 'a'
	 * @param a
	 * @param b
	 * @return
	 */
	public int checkDominance(Solution a, Solution b) {

		int flag1 = 0;
		int flag2 = 0;

		for (int i = 0; i < problem_.getNumberOfObjectives(); i++) {
			if (a.getObjective(i) < b.getObjective(i))
				flag1 = 1;
			else {
				if (a.getObjective(i) > b.getObjective(i))
					flag2 = 1;
			}
		}
		if (flag1 == 1 && flag2 == 0)
			return 1;
		else {
			if (flag1 == 0 && flag2 == 1)
				return -1;
			else
				return 0;
		}
	}
} // steady-state NSGA-II with 'ENLU'
