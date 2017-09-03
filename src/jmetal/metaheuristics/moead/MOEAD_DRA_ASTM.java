/**
 * MOEAD_DRA_ASTM.java
 * 
 * This is main implementation of MOEA/D-ASTM (with DRA).
 * 
 * Author:
 * 		Mengyuan Wu <mengyuan.wu@live.com>
 * 		Ke Li <k.li@exeter.ac.uk>
 * 
 * Affliation:
 * 		Department of Computer Science, City University of Hong Kong	
 * 		Department of Computer Science, University of Exeter
 * 
 * Reference:
 *		M. Wu, K. Li, S. Kwong, Y. Zhou, Q. Zhang,
 *		¡°Matching-Based Selection with Incomplete Lists for Decomposition Multi-Objective Optimization¡±,
 *		IEEE Transactions on Evolutionary Computation (TEVC), 21(4): 554¨C568, 2017. 
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

import jmetal.core.*;
import jmetal.util.JMException;
import jmetal.util.PseudoRandom;
import jmetal.util.comparators.DominanceComparator;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.InputStreamReader;
import java.util.*;

/**
 * Created by Mengyuan Wu on 23-Jun-15.
 */
public class MOEAD_DRA_ASTM extends Algorithm {
    private int populationSize_;
    /**
     * Stores the population
     */
    private SolutionSet population_;
    private SolutionSet union_;
    /**
     * Stores the values of the individuals
     */
    private Solution[] savedValues_;

    private double[] utility_;

    /**
     * Z vector (ideal point)
     */
    double[] z_;

    /**
     * NZ vector (nadir point)
     */
    double[] nz_;

    /**
     * Weight vectors
     */
    double[][] lambda_;

    /**
     * T: neighborhood size
     */
    int T_;

    /**
     * lmax_: maximum check for solution's preference list
     */
    int lmax_;

    /**
     * Neighborhood
     */
    int[][] neighborhood_;
    /**
     * delta: probability that parent solutions are selected from neighborhood
     */
    double delta_;
    /**
     * r: reduced preference list size
     */
    int r_;

    String functionType_;
    int evaluations_;
    /**
     * Operators
     */
    Operator crossover_;
    Operator mutation_;
    String crossoverName_  ;

    String dataDirectory_;

    /**
     * Constructor
     * @param problem Problem to solve
     */
    public MOEAD_DRA_ASTM(Problem problem) {
        super(problem);


    } // MOEA/D-DRA-ASTM

    public SolutionSet execute() throws JMException, ClassNotFoundException {

        int type;
        int maxEvaluations;

        String str1 = "EVA";
        String str2;

        evaluations_    = 0;
        dataDirectory_  = this.getInputParameter("dataDirectory").toString();
        maxEvaluations  = ((Integer) this.getInputParameter("maxEvaluations")).intValue();
        populationSize_ = ((Integer) this.getInputParameter("populationSize")).intValue();
        functionType_ = (String) this.getInputParameter("functionType");

        population_  = new SolutionSet(populationSize_);
        savedValues_ = new Solution[populationSize_];
        utility_     = new double[populationSize_];
        for (int i = 0; i < utility_.length; i++) {
            utility_[i] = 1.0;
        }

        T_ = ((Integer) this.getInputParameter("T")).intValue();
        lmax_ = ((Integer) this.getInputParameter("lmax")).intValue();
        delta_ = ((Double) this.getInputParameter("delta")).doubleValue();

        z_ 			  = new double[problem_.getNumberOfObjectives()];
        nz_ 		  = new double[problem_.getNumberOfObjectives()];
        lambda_ 	  = new double[populationSize_][problem_.getNumberOfObjectives()];
        neighborhood_ = new int[populationSize_][T_];

        crossover_ = operators_.get("crossover");
        mutation_  = operators_.get("mutation");
        crossoverName_ = (String) this.getInputParameter("crossoverName");

        // STEP 1. Initialization
        // STEP 1.1. Compute Euclidean distances between weight vectors and find T
        initUniformWeight();
        initNeighborhood();

        // STEP 1.2. Initialize population
        initPopulation();
        union_ = population_;

        // STEP 1.3. Initialize the ideal point 'z_' and the nadir point 'nz_'
        initIdealPoint();
        initNadirPoint();

        int iteration = 0;
        SolutionSet currentOffspring_;
        // STEP 2. Update
        do {
            // Select the satisfied subproblems
            List<Integer> order = tour_selection(10);
            currentOffspring_  = new SolutionSet(order.size());

            for (int i = 0; i < order.size(); i++) {
                int n = order.get(i);

                double rnd = PseudoRandom.randDouble();

                // STEP 2.1. Mating selection based on probability
                if (rnd < delta_)
                {
                    type = 1; // neighborhood
                } else {
                    type = 2; // whole population
                }
                Solution child;
                Solution[] parents;
                Vector<Integer> p = new Vector<Integer>();

                parents = matingSelection(p, n, 2, type);

                // Apply DE crossover and polynomial mutation
                if (crossoverName_.equals("DifferentialEvolutionCrossover")) {
                    child = (Solution) crossover_.execute(new Object[]{population_.get(n), parents});
                } else {
                    child = ((Solution[])crossover_.execute(new Solution[]{population_.get(n), parents[0]}))[0];
                }
                mutation_.execute(child);

                // Evaluation
                problem_.evaluate(child);
                evaluations_++;

				/* STEP 2.3. Update the ideal point 'z_' and nadir point 'nz_' */
                updateReference(child);
//                updateNadirPoint(child);

                // Add into the offspring population
                currentOffspring_.add(child);
            } // for

            // Combine the parent and the current offspring populations
            union_ = population_.union(currentOffspring_);
            updateNadirPoint();

            selection();

            // Update the utility value of subproblems
            iteration++;
            if (iteration % 30 == 0) {
                comp_utility();
            } 
        } while (evaluations_ <= maxEvaluations);

        return population_;
    }

    /**
     * Select the next parent population, based on the stable matching criteria
     */
    public void selection() {

        int[][]    solPref   = new int[union_.size()][];
        double[][] solMatrix = new double[union_.size()][];
        for (int i = 0; i < union_.size(); i++) {
            solPref[i]   = new int[populationSize_];
            solMatrix[i] = new double[populationSize_];
        }
        int[][]    subpPref   = new int[populationSize_][];
        double[][] subpMatrix = new double[populationSize_][];
        for (int i = 0; i < populationSize_; i++) {
            subpPref[i]   = new int[union_.size()];
            subpMatrix[i] = new double[union_.size()];
        }

        // Calculate the preference values of subproblem matrix and solution matrix
        for (int i = 0; i < union_.size(); i++) {
            for (int j = 0; j < populationSize_; j++) {
                subpMatrix[j][i] = fitnessFunction(union_.get(i), lambda_[j], functionType_);
                solMatrix[i][j]  = calculateDistance(union_.get(i), lambda_[j]);
            }
        }

        // Sort the preference value matrix to get the preference rank matrix
        for (int i = 0; i < populationSize_; i++) {
            for (int j = 0; j < union_.size(); j++)
                subpPref[i][j] = j;
            Utils.minFastSort(subpMatrix[i], subpPref[i], union_.size(), union_.size());
        }
        for (int i = 0; i < union_.size(); i++) {
            for (int j = 0; j < populationSize_; j++)
                solPref[i][j] = j;
            Utils.minFastSort(solMatrix[i], solPref[i], populationSize_, populationSize_);
        }

        int representativeSol[] = new int[populationSize_];
        for (int i = 0; i < populationSize_; i++) {
            double representativeF = Double.MAX_VALUE;
            representativeSol[i] = -1;
            for (int j = 0; j < union_.size(); j++) {
                if (solPref[j][0] == i) {
                    if (subpMatrix[i][j] < representativeF) {
                        representativeF = subpMatrix[i][j];
                        representativeSol[i] = j;
                    }
                }
            }
        }

        DominanceComparator dominanceCheck = new DominanceComparator();
        int solPreferListLengths[] = new int[union_.size()];
        for (int i = 0; i < union_.size(); i++) {
            solPreferListLengths[i] = problem_.getNumberOfObjectives();
            for (int l = solPreferListLengths[i] + 1; l <= lmax_; l++) {
                if (representativeSol[solPref[i][l - 1]] != -1) {
                    if (dominanceCheck.compare(union_.get(i), union_.get(representativeSol[solPref[i][l - 1]])) == -1) {
                        break;
                    }
                }
                solPreferListLengths[i] = l;
            }
        }

        StableMarriage smp = new StableMarriage(populationSize_, union_.size(), subpPref, solPref);
        int[] subpStatus = new int[populationSize_];
        int[] solStatus = new int[union_.size()];
        smp.stableMatchTwoLevel(subpStatus, solStatus, solPreferListLengths);

        for (int i = 0; i < populationSize_; i++)
            population_.replace(i, new Solution(union_.get(subpStatus[i])));
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
            vecInd[i] = (individual.getObjective(i) - z_[i]) / (nz_[i] - z_[i]);

        scale = innerproduct(vecInd, lambda) / innerproduct(lambda, lambda);
        for (int i = 0; i < problem_.getNumberOfObjectives(); i++)
            vecProj[i] = vecInd[i] - scale * lambda[i];

        distance = norm_vector(vecProj);

        return distance;
    }

    public double calculateDistanceDirect(double[] indivNorm, double[] lambda) {

        double scale;
        double distance;

        double[] vecProj = new double[problem_.getNumberOfObjectives()];

        scale = innerproduct(indivNorm, lambda) / innerproduct(lambda, lambda);
        for (int i = 0; i < problem_.getNumberOfObjectives(); i++)
            vecProj[i] = indivNorm[i] - scale * lambda[i];

        distance = norm_vector(vecProj);

        return distance;
    }

    /**
     * Calculate the aggregated distance between the solution and reference
     * line, by considering both of the perpendicular distance and projection
     * length
     *
     * @param individual
     * @param lambda
     * @return
     */
    public double calculateDistanceAggregate(Solution individual, double[] lambda) {
        double scale;
        double theta;
        double utility;
        double distance;

        double[] vecInd  = new double[problem_.getNumberOfObjectives()];
        double[] vecProj = new double[problem_.getNumberOfObjectives()];

        double nd = norm_vector(lambda);
        for (int i = 0; i < problem_.getNumberOfObjectives(); i++)
            lambda[i] = lambda[i] / nd;

        // vecInd has been normalized to the range [0,1]
        for (int i = 0; i < problem_.getNumberOfObjectives(); i++)
            vecInd[i] = (individual.getObjective(i) - z_[i]) / (nz_[i] - z_[i]);

        scale = innerproduct(vecInd, lambda) / innerproduct(lambda, lambda);
        for (int i = 0; i < problem_.getNumberOfObjectives(); i++)
            vecProj[i] = vecInd[i] - scale * lambda[i];

        distance = norm_vector(vecProj);

        theta = 0.5;

        utility = distance + theta * scale;

        return utility;
    }

    /**
     * Initialize the weight vectors for subproblems (We only use the data that are already available)
     */
    public void initUniformWeight() {
        if ((problem_.getNumberOfObjectives() == 2) && (populationSize_ < 100)) {
            lambda_[0][0] = 1.0;
            lambda_[0][1] = 0.0;
            lambda_[1][0] = 0.0;
            lambda_[1][1] = 1.0;
            for (int n = 2; n < populationSize_; n++) {
                double a = 1.0 * (n - 1) / (populationSize_ - 1);
                lambda_[n][0] = a;
                lambda_[n][1] = 1 - a;
            } // for
        } // if
        else {
            String dataFileName;
            dataFileName = "W" + problem_.getNumberOfObjectives() + "D_"
                    + populationSize_ + ".dat";

            try {
                // Open the file
                FileInputStream fis = new FileInputStream(dataDirectory_ + "/"
                        + dataFileName);
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
        }
    } // initUniformWeight

    /**
     * Compute the utility of subproblems
     */
    public void comp_utility() {

        double f1, f2, uti, delta;

        for (int i = 0; i < populationSize_; i++) {
            f1    = fitnessFunction(population_.get(i), lambda_[i], functionType_);
            f2 	  = fitnessFunction(savedValues_[i], lambda_[i], functionType_);

            delta = f2 - f1;
            if (delta > 0.001)
                utility_[i] = 1.0;
            else if (delta <= 0)
                utility_[i] = 0.95 * utility_[i];
            else {
                uti 		= (0.95 + (0.05 * delta / 0.001)) * utility_[i];
                utility_[i] = uti < 1.0 ? uti : 1.0;
            }
            savedValues_[i] = new Solution(population_.get(i));
        }
    }

    /**
     * Initialize the neighborhood of subproblems
     */
    public void initNeighborhood() {
        double[] x = new double[populationSize_];
        int[] idx = new int[populationSize_];

        for (int i = 0; i < populationSize_; i++) {
            // calculate the distances based on weight vectors
            for (int j = 0; j < populationSize_; j++) {
                x[j] = Utils.distVector(lambda_[i], lambda_[j]);
                // x[j] = dist_vector(population[i].namda,population[j].namda);
                idx[j] = j;
                // System.out.println("x["+j+"]: "+x[j]+
                // ". idx["+j+"]: "+idx[j]) ;
            } // for

            // find 'niche' nearest neighboring subproblems
            Utils.minFastSort(x, idx, populationSize_, T_);
            // minfastsort(x,idx,population.size(),niche);

            for (int k = 0; k < T_; k++) {
                neighborhood_[i][k] = idx[k];
                // System.out.println("neg["+i+","+k+"]: "+ neighborhood_[i][k])
                // ;
            }
        } // for
    } // initNeighborhood

    /**
     *
     */
    public void initPopulation() throws JMException, ClassNotFoundException {
        Solution newSolution;

        for (int i = 0; i < populationSize_; i++) {
            newSolution = new Solution(problem_);
            problem_.evaluate(newSolution);
            evaluations_++;
            population_.add(newSolution) ;
            savedValues_[i] = new Solution(newSolution);
        } // for
    } // initPopulation

    /**
     * Initialize the ideal point
     * @throws JMException
     * @throws ClassNotFoundException
     */
    void initIdealPoint() throws JMException, ClassNotFoundException {
        for (int i = 0; i < problem_.getNumberOfObjectives(); i++)
            z_[i] = 1.0e+30;

        for (int i = 0; i < populationSize_; i++)
            updateReference(population_.get(i));
    } // initIdealPoint

    /**
     * Initialize the nadir point
     * @throws JMException
     * @throws ClassNotFoundException
     */
    void initNadirPoint() throws JMException, ClassNotFoundException {
        updateNadirPoint();
    } // initNadirPoint

    /**
     * Mating selection is used to select the mating parents for offspring generation
     * @param list : the set of the indexes of selected mating parents
     * @param cid  : the id of current subproblem
     * @param size : the number of selected mating parents
     * @param type : 1 - neighborhood; otherwise - whole population
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

        return parents;
    } // matingSelection


    public List<Integer> tour_selection(int depth) {

        int i2, s2;
        int threshold;

        // selection based on utility
        List<Integer> selected  = new ArrayList<Integer>();
        List<Integer> candidate = new ArrayList<Integer>();

        for (int i = 0; i < problem_.getNumberOfObjectives(); i++)
            selected.add(i);

        // set of unselected weights
        for (int i = selected.size(); i < populationSize_; i++)
            candidate.add(i);

        threshold = (int) (populationSize_ / 5);
        while (selected.size() < threshold) {
            int best_idd = (int) (PseudoRandom.randDouble() * candidate.size());
            int best_sub = candidate.get(best_idd);

            for (int i = 1; i < depth; i++) {
                i2 = (int) (PseudoRandom.randDouble() * candidate.size());
                s2 = candidate.get(i2);
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
     * Update the ideal point, it is just an approximation with the best value for each objective
     *
     * @param individual
     */
    void updateReference(Solution individual) {
        for (int i = 0; i < problem_.getNumberOfObjectives(); i++) {
            if (individual.getObjective(i) < z_[i])
                z_[i] = individual.getObjective(i);
        }
    } // updateReference

    /**
     * Update the nadir point, it is just an approximation with worst value for each objective
     *
     * @param individual
     */
    void updateNadirPoint(Solution individual) {
        for (int i = 0; i < problem_.getNumberOfObjectives(); i++) {
            if (individual.getObjective(i) > nz_[i])
                nz_[i] = individual.getObjective(i);
        }
    } // updateNadirPoint

    /**
     * Update the nadir point, it is just an approximation with worst value for each objective
     *
     */
    void updateNadirPoint() {
        double intercepts[] = calculateIntercepts();
        for (int i = 0; i < problem_.getNumberOfObjectives(); i++) {
            nz_[i] = z_[i] + intercepts[i];
        }
    } // updateNadirPoint

    /**
     * Calculates the intercepts between the hyperplane formed by the extreme
     * points and each axis.  The original paper (1) is unclear how to handle
     * degenerate cases, which occurs more frequently at larger dimensions.  In
     * this implementation, we simply use the nadir point for scaling.
     *
     * @return an array of the intercept points for each objective
     */
    private double[] calculateIntercepts() {
        Solution[] extremePoints = extremePoints();
        boolean degenerate = false;
        double[] intercepts = new double[problem_.getNumberOfObjectives()];

        try {
            double[] b = new double[problem_.getNumberOfObjectives()];
            double[][] A = new double[problem_.getNumberOfObjectives()][problem_.getNumberOfObjectives()];

            for (int i = 0; i < problem_.getNumberOfObjectives(); i++) {

                b[i] = 1.0;

                for (int j = 0; j < problem_.getNumberOfObjectives(); j++) {
                    A[i][j] = extremePoints[i].getObjective(j) - z_[j];
                }
            }

            double[] result = lsolve(A, b);

            for (int i = 0; i < problem_.getNumberOfObjectives(); i++) {
                intercepts[i] = 1.0 / result[i];
            }
        } catch (RuntimeException e) {
            degenerate = true;
        }

        if (!degenerate) {
            // avoid small or negative intercepts
            for (int i = 0; i < problem_.getNumberOfObjectives(); i++) {
                if (intercepts[i] < 0.001) {
                    degenerate = true;
                    break;
                }
            }
        }

        if (degenerate) {
            Arrays.fill(intercepts, Double.NEGATIVE_INFINITY);

            for (int i = 0; i < union_.size(); i++) {
                for (int j = 0; j < problem_.getNumberOfObjectives(); j++) {
                    intercepts[j] = Math.max(Math.max(intercepts[j], 1e-10), union_.get(i).getObjective(j));
                }
            }
        }

        return intercepts;
    }

    // Gaussian elimination with partial pivoting
    // Copied from http://introcs.cs.princeton.edu/java/95linear/GaussianElimination.java.html
    /**
     * Gaussian elimination with partial pivoting.
     *
     * @param A the A matrix
     * @param b the b vector
     * @return the solved equation using Gaussian elimination
     */
    private double[] lsolve(double[][] A, double[] b) {
        int N = b.length;

        for (int p = 0; p < N; p++) {
            // find pivot row and swap
            int max = p;

            for (int i = p + 1; i < N; i++) {
                if (Math.abs(A[i][p]) > Math.abs(A[max][p])) {
                    max = i;
                }
            }

            double[] temp = A[p];
            A[p] = A[max];
            A[max] = temp;

            double t = b[p];
            b[p] = b[max];
            b[max] = t;

            // singular or nearly singular
            if (Math.abs(A[p][p]) <= 1e-10) {
                throw new RuntimeException("Matrix is singular or nearly singular");
            }

            // pivot within A and b
            for (int i = p + 1; i < N; i++) {
                double alpha = A[i][p] / A[p][p];
                b[i] -= alpha * b[p];

                for (int j = p; j < N; j++) {
                    A[i][j] -= alpha * A[p][j];
                }
            }
        }

        // back substitution
        double[] x = new double[N];

        for (int i = N - 1; i >= 0; i--) {
            double sum = 0.0;

            for (int j = i + 1; j < N; j++) {
                sum += A[i][j] * x[j];
            }

            x[i] = (b[i] - sum) / A[i][i];
        }

        return x;
    }

    /**
     * Returns the extreme points for all objectives.
     *
     * @return an array of extreme points, each index corresponds to each
     *         objective
     */
    private Solution[] extremePoints() {
        Solution[] result = new Solution[problem_.getNumberOfObjectives()];

        for (int i = 0; i < problem_.getNumberOfObjectives(); i++) {
            result[i] = findExtremePoint(i);
        }

        return result;
    }

    /**
     * Returns the extreme point in the given objective.  The extreme point is
     * the point that minimizes the achievement scalarizing function using a
     * reference point near the given objective.
     *
     * The NSGA-III paper (1) does not provide any details on the scalarizing
     * function, but an earlier paper by the authors (2) where some precursor
     * experiments are performed does define a possible function, replicated
     * below.
     *
     * @param objective the objective index
     * @return the extreme point in the given objective
     */
    private Solution findExtremePoint(int objective) {
        double eps = 0.000001;
        double[] weights = new double[problem_.getNumberOfObjectives()];

        for (int i = 0; i < problem_.getNumberOfObjectives(); i++) {
            if (i == objective) {
                weights[i] = 1.0;
            } else {
                weights[i] = eps;
            }
        }

        Solution result = null;
        double resultASF = Double.POSITIVE_INFINITY;

        for (int i = 0; i < union_.size(); i++) {
            Solution solution = union_.get(i);
            double solutionASF = fitnessFunction(solution, weights, "_TCH2");

            if (solutionASF < resultASF) {
                result = solution;
                resultASF = solutionASF;
            }
        }

        return result;
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

    /**
     * Calculate the fitness value of a given individual, based on the specific scalarizing function
     * @param individual
     * @param lambda
     * @return
     */
    double fitnessFunction(Solution individual, double[] lambda, String type) {
        double fitness;
        fitness = 0.0;

        if (type.equals("_TCH2")) {
            double maxFun = -1.0e+30;

            for (int i = 0; i < problem_.getNumberOfObjectives(); i++) {
                double diff;
                diff = Math.abs((individual.getObjective(i) - z_[i]) / (nz_[i] - z_[i]));

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
        } else if (type.equals("_PBI")) {
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
                realA[n] = (individual.getObjective(n) - z_[n]) / (nz_[n] - z_[n]);

            // distance along the line segment
            double d1 = Math.abs(innerproduct(realA, lambda));

            // distance to the line segment
            for (int n = 0; n < problem_.getNumberOfObjectives(); n++)
                realB[n] = realA[n] - d1 * lambda[n];
            double d2 = norm_vector(realB);

            fitness = d1 + theta * d2;
        } else {
            System.out.println("MOEAD.fitnessFunction: unknown type "
                    + functionType_);
            System.exit(-1);
        }
        return fitness;
    } // fitnessEvaluation

}
