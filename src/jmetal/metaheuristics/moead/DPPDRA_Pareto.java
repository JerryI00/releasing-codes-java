/**
 * DPPDRA_Pareto.java
 * 
 * This is main implementation of ND/DPP-DRA.
 * 
 * Author:
 * 		Ke Li <k.li@exeter.ac.uk>
 * 
 * Affliation:
 * 		Department of Computer Science, University of Exeter
 * 
 * Reference:
 *		K. Li, S. Kwong, K. Deb,
 *		¡°A Dual Population Paradigm for Evolutionary Multiobjective Optimization¡±, 
 *		Information Sciences (INS). 309: 50¨C72, 2015. 
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

package jmetal.metaheuristics.moead;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.LinkedList;
import java.util.List;
import java.util.StringTokenizer;
import java.util.Vector;

import jmetal.core.Algorithm;
import jmetal.core.Operator;
import jmetal.core.Problem;
import jmetal.core.Solution;
import jmetal.core.SolutionSet;
import jmetal.core.Variable;
import jmetal.util.Distance;
import jmetal.util.JMException;
import jmetal.util.PseudoRandom;
import jmetal.util.Ranking;
import jmetal.util.comparators.CrowdingComparator;

public class DPPDRA_Pareto extends Algorithm {
	
	private int populationSize_;
	
	private SolutionSet population_;	// evolutionary population
	private SolutionSet archive_pop;	// archive population
	private SolutionSet mixed_pop;		// mixed population of archive and evolutionary populations
	private SolutionSet final_pop;      // final output population
	private Solution[] savedValues_;	// Stores the values of the individuals

	double[][] lambda_;		// Lambda vectors
	
	int T_;					// neighborhood size
	int evaluations_;		// counter of evaluation times
	double delta_;			// probability that parent solutions are selected from neighborhood
	int[][] neighborhood_;	// neighborhood structure
	
	String functionType_;
	
	private double [] utility_;
	
	Operator crossover_;
	Operator mutation_;

	String dataDirectory_;
	
	/***********************************************************************************/		
	double[] zp_;	// ideal point for Pareto-based population
	double[] zd_;	// ideal point for decomposition-based archive
	double[] nzp_;	// nadir point for Pareto-based population
	double[] nzd_;	// nadir point for decomposition-based population
	
	int[][] subregionMatrix_;	// subregion occupation record
	
	/** 
	 * Constructor
	 * 
	 * @param problem: Problem to solve
	 */
	public DPPDRA_Pareto(Problem problem) {
		super(problem);

//		functionType_ = "_TCHE1";
		functionType_ = "_TCHE2";

	} // DMOEA

	public SolutionSet execute() throws JMException, ClassNotFoundException {

		int maxEvaluations;

		evaluations_ = 0;
				
		crossover_ = operators_.get("crossover");
		mutation_  = operators_.get("mutation");
		
		dataDirectory_  = this.getInputParameter("dataDirectory").toString();
		maxEvaluations  = ((Integer) this.getInputParameter("maxEvaluations")).intValue();
		populationSize_ = ((Integer) this.getInputParameter("populationSize")).intValue();
		
		T_ = 20;
	    delta_ = 0.9;
		
		population_  = new SolutionSet(populationSize_);
		archive_pop  = new SolutionSet(10 * populationSize_);

		savedValues_  = new Solution[populationSize_];
		neighborhood_ = new int[populationSize_][T_];
		lambda_ 	  = new double[populationSize_][problem_.getNumberOfObjectives()];
		
		utility_ = new double[populationSize_];
		for (int i = 0; i < utility_.length; i++)
			utility_[i] = 1.0;
		
		zp_  = new double[problem_.getNumberOfObjectives()];	// ideal point for Pareto-based population
		zd_  = new double[problem_.getNumberOfObjectives()];	// ideal point for decomposition-based population
		nzp_ = new double[problem_.getNumberOfObjectives()];	// nadir point for Pareto-based population
		nzd_ = new double[problem_.getNumberOfObjectives()];	// nadir point for decomposition-based archive

		subregionMatrix_ = new int[populationSize_][populationSize_];
		for (int i = 0; i < populationSize_; i++)
			subregionMatrix_[i][i] = 1;
		
		// STEP 1. Initialization
		initUniformWeight();
		initNeighborhood();
		initPopulation();
		initIdealPoint();
		initNadirPoint();

		int gen = 0;
		// STEP 2. Update
		do {
			int[] permutation = new int[populationSize_];
			Utils.randomPermutation(permutation, populationSize_);

			List<Integer> order = tour_selection(10);

			for (int i = 0; i < order.size(); i++) {
				int n = order.get(i);

				int type;
				double rnd = PseudoRandom.randDouble();

				// STEP 2.1. Mating selection based on probability
				if (rnd < delta_) // if (rnd < realb)
					type = 1; // neighborhood
				else
					type = 2; // whole population
				Solution child;
				Solution[] parents = new Solution[3];
				Vector<Integer> p  = new Vector<Integer>();
				parents = matingSelection(p, n, 2, type);

				// STEP 2.2. Reproduction
				// Apply DE crossover
				child = (Solution) crossover_.execute(new Object[] {population_.get(n), parents });

				// Apply mutation
				mutation_.execute(child);

				// Evaluation
				problem_.evaluate(child);
				evaluations_++;

				// STEP 2.4. Update ideal and nadir points
				updateReference(child, zp_);
				updateReference(child, zd_);
				updateNadirPoint(child, nzp_);
				updateNadirPoint(child, nzd_);
				
				// STEP 2.5. Update of solutions
				updateProblem(child);
				updateArchive(child);
			} // for

			gen++;
			if (gen % 30 == 0)
				comp_utility();

		} while (evaluations_ < maxEvaluations);
		
		final_pop = filtering(populationSize_);
		
		return final_pop;
	}

	/**
	 * Initialize the weight vectors for subproblems (We only use the data that are already available)
	 */
	public void initUniformWeight() {
		String dataFileName;
		dataFileName = "W" + problem_.getNumberOfObjectives() + "D_" + populationSize_ + ".dat";

		try {
			// Open the file
			FileInputStream fis = new FileInputStream(dataDirectory_ + "/" + dataFileName);
			InputStreamReader isr = new InputStreamReader(fis);
			BufferedReader br = new BufferedReader(isr);

			int i = 0;
			int j = 0;
			String aux = br.readLine();
			while (aux != null) {
				StringTokenizer st = new StringTokenizer(aux);
				j = 0;
				while (st.hasMoreTokens()) {
					double value = (new Double(st.nextToken())).doubleValue();
					lambda_[i][j] = value;
					j++;
				}
				aux = br.readLine();
				i++;
			}
			br.close();
		} catch (Exception e) {
			System.out
					.println("initUniformWeight: failed when reading for file: "
							+ dataDirectory_ + "/" + dataFileName);
			e.printStackTrace();
		}
	} // initUniformWeight
	
	/**
	 * Initialize the neighborhood structure
	 */
	public void initNeighborhood() {
		double[] x = new double[populationSize_];
		int[] idx = new int[populationSize_];

		for (int i = 0; i < populationSize_; i++) {
			// calculate the distances based on weight vectors
			for (int j = 0; j < populationSize_; j++) {
				x[j] = Utils.distVector(lambda_[i], lambda_[j]);
				idx[j] = j;
			} // for

			// find 'niche' nearest neighboring subproblems
			Utils.minFastSort(x, idx, populationSize_, T_);
			for (int k = 0; k < T_; k++) {
				neighborhood_[i][k] = idx[k];
			}
		} // for
	} // initNeighborhood

	/**
	 * Initialize the population
	 * 
	 * @throws JMException
	 * @throws ClassNotFoundException
	 */
	public void initPopulation() throws JMException, ClassNotFoundException {
		for (int i = 0; i < populationSize_; i++) {
			Solution newSolution = new Solution(problem_);
			problem_.evaluate(newSolution);
			evaluations_++;

			population_.add(newSolution);

			newSolution.setRegion(i);
			archive_pop.add(newSolution);
			addIndiv(newSolution, i);
			
			savedValues_[i] = new Solution(newSolution);
		}
	} // initPopulation

	/**
	 * Initialize the ideal objective vector
	 * @throws JMException
	 * @throws ClassNotFoundException
	 */
	void initIdealPoint() throws JMException, ClassNotFoundException {
		for (int i = 0; i < problem_.getNumberOfObjectives(); i++)
			zp_[i] = 1.0e+30;

		for (int i = 0; i < populationSize_; i++)
			updateReference(population_.get(i), zp_);
		
		zd_ = zp_;
	} // initIdealPoint
	
	/**
	 * Initialize the nadir point
	 * 
	 * @throws JMException
	 * @throws ClassNotFoundException
	 */
	void initNadirPoint() throws JMException, ClassNotFoundException {
		for (int i = 0; i < problem_.getNumberOfObjectives(); i++)
			nzp_[i] = -1.0e+30;

		for (int i = 0; i < populationSize_; i++)
			updateNadirPoint(population_.get(i), nzp_);
		
		nzd_ = nzp_;
	} // initNadirPoint
	
	/**
	 * compute the utility of each subproblem
	 */
	public void comp_utility() {
		double f1, f2, uti, delta;

		for (int i = 0; i < populationSize_; i++) {
			f1 = fitnessFunction(population_.get(i), lambda_[i]);
			f2 = fitnessFunction(savedValues_[i], lambda_[i]);

			delta = f2 - f1;
			if (delta > 0.001)
				utility_[i] = 1.0;
			else {
				uti = 0.95 * (1.0 + delta / 0.001) * utility_[i];
				utility_[i] = uti < 1.0 ? uti : 1.0;
			}
			savedValues_[i] = new Solution(population_.get(i));
		}
	}

	/**
	 * Restricted mating selection: select mating parents
	 * 
	 * @param list: the set of the indexes of selected mating parents
	 * @param cid : the id of current subproblem
	 * @param size: the number of selected mating parents
	 * @param type: 1 - neighborhood; otherwise - whole population
	 */
	public Solution[] matingSelection(Vector<Integer> list, int cid, int size, int type) {
		
		int ss, r, p;
		
		Solution[] parents = new Solution[3];

		ss = neighborhood_[cid].length;
		while (list.size() < size) {
			if (type == 1) {
				r = PseudoRandom.randInt(0, ss - 1);
				p = neighborhood_[cid][r];
			} else {
				p = PseudoRandom.randInt(0, populationSize_ - 1);
			}
			boolean flag = true;
			for (int i = 0; i < list.size(); i++) {
				if (list.get(i) == p) // p is in the list
				{
					flag = false;
					break;
				}
			}

			if (flag) {
				list.addElement(p);
			}
		}
		
		parents[0] = population_.get(list.get(0));
		parents[1] = population_.get(list.get(1));
		parents[2] = population_.get(cid);
		
		int idx = countSols(list.get(0)); 
		if (idx != -1)
			parents[0] = archive_pop.get(idx);
		
		return parents;
	} // matingSelection
	
	/**
	 * Update the ideal objective vector
	 * @param indiv
	 */
	void updateReference(Solution indiv, double[] z_) {
		for (int i = 0; i < problem_.getNumberOfObjectives(); i++) {
			if (indiv.getObjective(i) < z_[i]) {
				z_[i] = indiv.getObjective(i);
			}
		}
	} // updateReference
	
	/**
  	 * Update the nadir point
  	 * 
  	 * @param individual
  	 */
	void updateNadirPoint(Solution individual, double[] nz_) {
		for (int i = 0; i < problem_.getNumberOfObjectives(); i++) {
			if (individual.getObjective(i) > nz_[i])
				nz_[i] = individual.getObjective(i);
		}
	} // updateNadirPoint

	/**
	 * Update the convergence archive (use the procedure of NSGA-II)
	 * 
	 * @param Child
	 * @param population
	 */
	public void updateArchive(Solution indiv) {

		Distance distance = new Distance();

		// set the location of 'indiv' in the Pareto-based archive
		setLocation(indiv, zp_, nzp_);

		if (archive_pop.size() > 0) {
			int flag = 0;
			for (int i = 0; i < archive_pop.size(); i++) {
				Solution Parent = archive_pop.get(i);
				switch (checkDominance(indiv, Parent)) {
				case 1: { // child dominates one parent
					flag = 1;
					archive_pop.replace(i, indiv);
					deleteIndiv(Parent, i);
					addIndiv(indiv, i);
					return;
				}
				case -1: { // child is dominated by a parent
					return;
				}
				case 0: { // both child and the current parent is non-dominated
					break;
				}
				}
			}
			if (flag == 0) {
				// truncation by random replacement
//				Random r = new Random();
//				int replace_id = r.nextInt(populationSize_);
//
//				Solution Parent = archive_pop.get(replace_id);
//
//				archive_pop.replace(replace_id, indiv);
//				deleteIndiv(Parent);
//				addIndiv(indiv);

				// truncation by crowding distance
				 archive_pop.add(indiv);
				
				 distance.crowdingDistanceAssignment(archive_pop, problem_.getNumberOfObjectives());
				 archive_pop.sort(new CrowdingComparator());
				
				 Solution tempSol = archive_pop.get(archive_pop.size() - 1);

				 int idx = archive_pop.size() - 1;
				 if (!tempSol.equals(indiv)) {
					 archive_pop.remove(idx);
					 int prevLocation = tempSol.readRegion();
					 int curLocation  = indiv.readRegion();
					 
					 deleteIndiv(tempSol, prevLocation);
					 addIndiv(indiv, curLocation);
				 } else {
					 archive_pop.remove(idx);
				 }
			}
		}
	}
	
	/**
	 * update the decomposition-based archive
	 * 
	 * @param indiv
	 * 
	 */
	void updateProblem(Solution indiv) {

		double cur_func, prev_func;

		setLocation(indiv, zd_, nzd_);

		int location = indiv.readRegion();
		
		Solution prevSol = population_.get(location);
		cur_func  = fitnessFunction(indiv, lambda_[location]);
		prev_func = fitnessFunction(prevSol, lambda_[location]);
		if (cur_func < prev_func)
			population_.replace(location, indiv);

	} // updateProblem
	
	/**
	 * Find a solution from the specific subregion in the Pareto-based archive
	 * 
	 * @param idx
	 * @return
	 */
	public int countSols(int idx) {
		
		Vector<Integer> list = new Vector<Integer>();
		
		for (int i = 0; i < populationSize_; i++) {
			if (subregionMatrix_[idx][i] == 1)
				list.addElement(i);
		}
		
		if (list.size() == 0)
			return -1;
		else 
			return list.get(PseudoRandom.randInt(0, list.size() - 1));
	}
	
	/**
	 * Set the location of a solution based on the orthogonal distance
	 * 
	 * @param indiv
	 */
	public void setLocation(Solution indiv, double[] z_, double[] nz_) {

		int minIdx;
		double distance, minDist;

		minIdx = 0;
		distance = calculateDistance2(indiv, lambda_[0], z_, nz_);
		minDist = distance;
		for (int i = 1; i < populationSize_; i++) {
			distance = calculateDistance2(indiv, lambda_[i], z_, nz_);
			if (distance < minDist) {
				minIdx = i;
				minDist = distance;
			}
		}

		indiv.setRegion(minIdx);
		indiv.Set_associateDist(minDist);
	}

	/**
	 * delete 'indiv' from the subregion record
	 * 
	 * @param indiv
	 * @param idx
	 */
	public void deleteIndiv(Solution indiv, int idx) {

		int location = indiv.readRegion();
		subregionMatrix_[location][idx] = 0;
	}

	/**
	 * add 'indiv' to the subregion record
	 * @param indiv
	 * @param idx
	 */
	public void addIndiv(Solution indiv, int idx) {

		int location = indiv.readRegion();
		subregionMatrix_[location][idx] = 1;
	}
	
	double fitnessFunction(Solution indiv, double[] lambda) {
		double fitness;
		fitness = 0.0;
	
		if (functionType_.equals("_TCHE1")) {
			double maxFun = -1.0e+30;
	
			for (int i = 0; i < problem_.getNumberOfObjectives(); i++) {
				double diff = Math.abs(indiv.getObjective(i) - zd_[i]);
	
				double feval;
				if (lambda[i] == 0) {
					feval = 0.0001 * diff;
				} else {
					feval = diff * lambda[i];
				}
				if (feval > maxFun) {
					maxFun = feval;
				}
			} // for
	
			fitness = maxFun;
		} else if (functionType_.equals("_TCHE2")) {
			double maxFun = -1.0e+30;
	
			for (int i = 0; i < problem_.getNumberOfObjectives(); i++) {
				double diff = Math.abs(indiv.getObjective(i) - zd_[i]);
	
				double feval;
				if (lambda[i] == 0) {
					feval = diff / 0.000001;
				} else {
					feval = diff / lambda[i];
				}
				if (feval > maxFun) {
					maxFun = feval;
				}
			} // for
			fitness = maxFun;
			} else if(functionType_.equals("_OO")) {
				for (int i = 0; i < problem_.getNumberOfObjectives(); i++) {
					fitness += indiv.getObjective(i);
			}
		} else if (functionType_.equals("_PBI")) {
			double theta; // penalty parameter
			theta = 5.0;

			// normalize the weight vector (line segment)
			double nd = norm_vector(lambda);
			for (int i = 0; i < problem_.getNumberOfObjectives(); i++)
				lambda[i] = lambda[i] / nd;

			double[] realA = new double[problem_.getNumberOfObjectives()];
			double[] realB = new double[problem_.getNumberOfObjectives()];

			// difference between current point and reference point
			for (int n = 0; n < problem_.getNumberOfObjectives(); n++)
				realA[n] = (indiv.getObjective(n) - zd_[n]);

			// distance along the line segment
			double d1 = Math.abs(innerproduct(realA, lambda));

			// distance to the line segment
			for (int n = 0; n < problem_.getNumberOfObjectives(); n++)
				realB[n] = (indiv.getObjective(n) - (zd_[n] + d1 * lambda[n]));
			double d2 = norm_vector(realB);

			// fitness = d2;
			fitness = d1 + theta * d2;
			}else {
			System.out.println("MOEAD.fitnessFunction: unknown type "
					+ functionType_);
			System.exit(-1);
		}
		return fitness;
	} // fitnessEvaluation
	
	public List<Integer> tour_selection(int depth) {

		// selection based on utility
		List<Integer> selected = new ArrayList<Integer>();
		List<Integer> candidate = new ArrayList<Integer>();

		for (int i = 0; i < problem_.getNumberOfObjectives(); i++)
			selected.add(i); // WARNING! HERE YOU HAVE TO USE THE WEIGHT
								// PROVIDED BY QINGFU (NOT SORTED!!!!)

		for (int i = problem_.getNumberOfObjectives(); i < populationSize_; i++)
			candidate.add(i); // set of unselected weights

		while (selected.size() < (int) (populationSize_ / 5.0)) {
			// int best_idd = (int) (rnd_uni(&rnd_uni_init)*candidate.size()),
			// i2;
			int best_idd = (int) (PseudoRandom.randDouble() * candidate.size());
			// System.out.println(best_idd);
			int i2;
			int best_sub = candidate.get(best_idd);
			int s2;
			for (int i = 1; i < depth; i++) {
				i2 = (int) (PseudoRandom.randDouble() * candidate.size());
				s2 = candidate.get(i2);
				// System.out.println("Candidate: "+i2);
				if (utility_[s2] > utility_[best_sub]) {
					best_idd = i2;
					best_sub = s2;
				}
			}
			selected.add(best_sub);
			candidate.remove(best_idd);
		}
		return selected;
	}
	
	/**
	 * Calculate the perpendicular distance between the solution and reference
	 * line
	 * 
	 * @param individual
	 * @param lambda
	 * @return
	 */
	public double calculateDistance(Solution individual, double[] lambda) {
		double scale;
		double distance;

		double[] vecInd  = new double[problem_.getNumberOfObjectives()];
		double[] vecProj = new double[problem_.getNumberOfObjectives()];
		
		// vecInd has been normalized to the range [0,1]
		for (int i = 0; i < problem_.getNumberOfObjectives(); i++)
			vecInd[i] = (individual.getObjective(i) - zd_[i]) / (nzd_[i] - zd_[i]);

		scale = innerproduct(vecInd, lambda) / innerproduct(lambda, lambda);
		for (int i = 0; i < problem_.getNumberOfObjectives(); i++)
			vecProj[i] = vecInd[i] - scale * lambda[i];

		distance = norm_vector(vecProj);
		
		return distance;
	}
	
	public double calculateDistance2(Solution indiv, double[] lambda, double[] z_, double[] nz_) {

		// normalize the weight vector (line segment)
		double nd = norm_vector(lambda);
		for (int i = 0; i < problem_.getNumberOfObjectives(); i++)
			lambda[i] = lambda[i] / nd;

		double[] realA = new double[problem_.getNumberOfObjectives()];
		double[] realB = new double[problem_.getNumberOfObjectives()];

		// difference between current point and reference point
		for (int i = 0; i < problem_.getNumberOfObjectives(); i++)
			realA[i] = (indiv.getObjective(i) - z_[i]);

		// distance along the line segment
		double d1 = Math.abs(innerproduct(realA, lambda));

		// distance to the line segment
		for (int i = 0; i < problem_.getNumberOfObjectives(); i++)
			realB[i] = (indiv.getObjective(i) - (z_[i] + d1 * lambda[i]));

		double distance = norm_vector(realB);

		return distance;
	}
	
	public int checkDominance(Solution a, Solution b) {
		
		int flag1 = 0;
		int flag2 = 0;

		for (int i = 0; i < problem_.getNumberOfObjectives(); i++) {
			if (a.getObjective(i) < b.getObjective(i)) {
				flag1 = 1;
			} else {
				if (a.getObjective(i) > b.getObjective(i)) {
					flag2 = 1;
				}
			}
		}
		if (flag1 == 1 && flag2 == 0) {
			return 1;
		} else {
			if (flag1 == 0 && flag2 == 1) {
				return -1;
			} else {
				return 0;
			}
		}
	}
	
	/**
	 * Calculate the dot product of two vectors
	 * 
	 * @param vec1
	 * @param vec2
	 * @return
	 */
	public double innerproduct(double[] vec1, double[] vec2) {
		double sum = 0;
		
		for (int i = 0; i < vec1.length; i++)
			sum += vec1[i] * vec2[i];
		
		return sum;
	}

	/**
	 * Calculate the norm of the vector
	 * @param z
	 * @return
	 */
	public double norm_vector(double[] z) {
		double sum = 0;
		
		for (int i = 0; i < problem_.getNumberOfObjectives(); i++)
			sum += z[i] * z[i];
		
		return Math.sqrt(sum);
	}
	
	public SolutionSet filtering(int final_size) {
		int popsize = populationSize_ + archive_pop.size();
		
		mixed_pop = new SolutionSet(popsize);
		for (int i = 0; i < populationSize_; i++) {
			mixed_pop.add(population_.get(i));
		}
		for (int i = 0; i < archive_pop.size(); i++) {
			mixed_pop.add(archive_pop.get(i));
		}
		
		return mixed_pop;
	}

} // MOEAD

