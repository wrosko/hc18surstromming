import numpy as np
import pickle
import matplotlib.pyplot as plt
import multiprocessing as mp


input_names = ["a_example",
               "b_should_be_easy",
               "c_no_hurry",
               "d_metropolis",
               "e_high_bonus"]


class Myrorna:


    def __init__(self, prob_name, K, alpha, beta, gamma, rho):
        self.cpus = mp.cpu_count() - 1
        self.prob_name = prob_name
        self.K = K*self.cpus
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.rho = rho
        self.load_pickle(prob_name)
        self.max_score = self.lengths_of_rides.sum() + self.bonus*len(self.lengths_of_rides)  # theoretical maximum score -- in general, there will not be possible solutions that can achieve this
        print("Max score:", self.max_score)
        self.min_tstep = int(np.min(self.lengths_of_rides[1:]) - 1)
        print("Min ride length:", self.min_tstep+1)
        self.max_tstep = int(np.max(self.lengths_of_rides[1:]) - 1)
        print("Max ride length:", self.max_tstep+1)
        self.slocs = np.zeros(self.N)


    def load_pickle(self, prob_name):
        with open("inputs/"+prob_name+".pickle", "rb") as f:
            self.N = pickle.load(f)
            self.dist_matrix = pickle.load(f)
            self.lengths_of_rides = pickle.load(f)
            self.earliest_starts = pickle.load(f)
            self.latest_starts = pickle.load(f)
            self.bonus = pickle.load(f)
            self.T = pickle.load(f)
        print("loaded pickle for "+prob_name)



    def write_solution_to_file(self, paths, fname):

        with open(fname, "w") as f:
            for i,path in enumerate(paths):

                outstr = str(len(path)-1)
                for stop in path[1:]:
                    outstr += " {:d}".format(int(stop-1))
                outstr += "\n"

                f.write(outstr)

        print("solution written to file "+fname)


    def calc_score(self, t, current_pos, next_pos=None):
        tb4s = self.time_b4_start(t, next_pos)
        tb4e = self.time_b4_end(t, next_pos)
        dij = self.dist_matrix[current_pos, next_pos]

        earns_bonus = np.greater_equal(tb4s, dij) # if it takes less time to get there than the time before it starts
        not_too_late = np.greater_equal(tb4e, dij) # " " " " " " " " " " ends

        return (not_too_late * (self.lengths_of_rides[next_pos] + self.bonus * earns_bonus)) # zero where opportunity is past


    def time_b4_start(self, t, pos=None):
        return self.earliest_starts[pos] - t


    def time_b4_end(self, t, pos=None):
        return self.latest_starts[pos] - t


    def time_busy(self, t, current_pos, next_pos=None):
        tb4s = self.time_b4_start(t, next_pos)

        return np.maximum(self.dist_matrix[current_pos, next_pos], tb4s) + self.lengths_of_rides[next_pos] # if we arrive too early, we still have to wait until the start


    def calc_eta(self, t, not_tabu, current_pos, next_pos=None):

        score = self.calc_score(t, current_pos, next_pos)
        not_excluded = np.array(
                     [x in not_tabu for x in range(len(self.lengths_of_rides))])

        #numer1 = np.power(score, 2) * not_excluded  # zero if pos is in tabu list
        numer1 = score * not_excluded  # zero if pos is in tabu list

        if numer1.sum()==0:
            return 0, True

        tb4s = self.time_b4_start(t, next_pos)
        #tb4e = self.time_b4_end(t, next_pos)
        dij = self.dist_matrix[current_pos, next_pos]

        denom1 = 1 + dij # scale by time taken to get to start
        #denom2 = 1 + np.maximum(0, tb4s - dij) # scale by time waiting
        denom2 = 1 + np.absolute(tb4s - dij) # scale by time wasted
        #denom3 = 1 + (tb4e - tb4s) # scale by size of window of opportunity

        #return (numer1/(denom1*denom2*denom3)), False
        return (numer1/(denom1*denom2)), False


    def choose_next_step(self, tau, eta):
        r = np.random.rand()
        p = np.power(tau, self.alpha) * np.power(eta, self.beta)
        intervals = np.cumsum(p)/p.sum()  # essentially roulette-wheel selection
        return int(np.where(r - intervals <= 0)[0][0])


    def init_trial(self):
        paths = [[0] for _ in range(self.N)]  # store path taken per traveller
        scores = [[0] for _ in range(self.N)]  # store score for each step of each traveller
        avail = np.zeros(self.N, dtype=np.uint32)  # stores time at which each traveller becomes available
        locs = self.slocs.copy() # records locations of travellers during their travels
        tabu = set(locs)
        not_tabu = set(range(self.dist_matrix.shape[0])) - tabu  # store set of nodes that are still available to travellers
        return paths, scores, avail, locs, tabu, not_tabu


    def single_trial_stochastic(self, tau_matrix, greedy=False):
        if greedy:
            gamma = 1
        else:
            gamma = self.gamma

        paths, scores, avail, locs, tabu, not_tabu = self.init_trial()
        blacklist = set()

        lnt = len(not_tabu)  # keep track of number of available nodes -- if it stagnates, exit loop and return
        clnt = 0
        while clnt < self.max_tstep/self.min_tstep and len(blacklist) < self.N and lnt > 0:
            last_lnt = len(not_tabu)

            travellers = set(np.random.choice(self.N, size=int(gamma*self.N), replace=False)) # select some number of travellers randomly
            travellers -= blacklist  # remove blacklisted travellers

            for trv in travellers:  # give each traveller a chance to take a step

                t = avail[trv]
                current_pos = int(locs[trv])

                eta, cont_flag = self.calc_eta(t, not_tabu, current_pos)  # get visibility of each node
                if cont_flag:  # if no nodes are available for whatever reason, skip this traveller going forward (big speedup with this!)
                    blacklist.add(trv)
                    continue

                if greedy:
                    next_pos = int(np.argmax(eta))  # greedy search
                else:
                    tau = tau_matrix[current_pos,:]
                    next_pos = self.choose_next_step(tau, eta)  # select next step probabilistically according to ACO algorithm

                busy = self.time_busy(t, current_pos, next_pos=next_pos)  # the time that the traveller is occupied before becoming available again
                avail[trv] += int(busy)

                paths[trv].append(next_pos)
                locs[trv] = next_pos

                score = self.calc_score(t, current_pos, next_pos=next_pos)  # calculate score attained by step
                scores[trv].append(score)

                not_tabu.remove(next_pos)
                tabu.add(next_pos)
                lnt = len(not_tabu)

                if lnt==0:
                    break

            if lnt==last_lnt:  # if no nodes were selected in last pass
                clnt += 1
            else:
                clnt = 0

        return paths, scores


    def single_trial_sequential(self, tau_matrix, greedy=False):

        paths, scores, avail, locs, tabu, not_tabu = self.init_trial()

        for t in range(self.T):

            travellers = np.where(avail <= t)[0]
            if len(travellers) > 1:
                np.random.shuffle(travellers) # select all travellers who are available now and randomize the order they can make their choice

            for trv in travellers:  # give each traveller a chance to take a step

                current_pos = int(locs[trv])

                eta, cont_flag = self.calc_eta(t, not_tabu, current_pos)  # get visibility of each node
                if cont_flag:  # if no nodes are available for whatever reason, skip this traveller
                    continue

                if greedy:
                    next_pos = int(np.argmax(eta))  # greedy search
                else:
                    tau = tau_matrix[current_pos,:]
                    next_pos = self.choose_next_step(tau, eta)  # select next step probabilistically according to ACO algorithm

                busy = self.time_busy(t, current_pos, next_pos=next_pos)  # the time that the traveller is occupied before becoming available again
                avail[trv] += int(busy)

                paths[trv].append(next_pos)
                locs[trv] = next_pos

                score = self.calc_score(t, current_pos, next_pos=next_pos)  # calculate score attained by step
                scores[trv].append(score)

                not_tabu.remove(next_pos)
                tabu.add(next_pos)

        return paths, scores


    def single_trial_hybrid(self, tau_matrix, greedy=False):

        if greedy:
            gamma = 1
        else:
            gamma = self.gamma

        paths, scores, avail, locs, tabu, not_tabu = self.init_trial()

        tstep = self.min_tstep

        lnt = len(not_tabu)  # keep track of number of available nodes -- if it stagnates, exit loop and return
        clnt = 0

        while clnt < self.max_tstep/self.min_tstep and lnt > 0:
            last_lnt = len(not_tabu)

            travellers = np.where(avail <= tstep)[0]
            if len(travellers) > 0:
                trvwgts = 1/(avail[travellers]+1)  # weight such that earlier availabilities are chosen more often
                travellers = np.random.choice(travellers, int(gamma*len(travellers)), replace=False, p=trvwgts/trvwgts.sum()) # select all travellers who are available now and randomize the order they can make their choice

            for trv in travellers:  # give each traveller a chance to take a step

                t = avail[trv]

                current_pos = int(locs[trv])

                eta, cont_flag = self.calc_eta(t, not_tabu, current_pos)  # get visibility of each node
                if cont_flag:  # if no nodes are available for whatever reason, skip this traveller and set availability to past the deadline
                    avail[trv] = self.T
                    continue

                if greedy:
                    next_pos = int(np.argmax(eta))  # greedy search
                else:
                    tau = tau_matrix[current_pos,:]
                    next_pos = self.choose_next_step(tau, eta)  # select next step probabilistically according to ACO algorithm

                busy = self.time_busy(t, current_pos, next_pos=next_pos)  # the time that the traveller is occupied before becoming available again
                avail[trv] += int(busy)

                paths[trv].append(next_pos)
                locs[trv] = next_pos

                score = self.calc_score(t, current_pos, next_pos=next_pos)  # calculate score attained by step
                scores[trv].append(score)

                not_tabu.remove(next_pos)
                tabu.add(next_pos)
                lnt = len(not_tabu)

                if lnt==0:
                    break

            if lnt==last_lnt:  # if no nodes were selected in last pass
                clnt += 1
            else:
                clnt = 0

            tstep += self.min_tstep

        return paths, scores


    def total_score(self, scores):
        return sum([sum(per_trv) for per_trv in scores])


    def calc_dtau(self, paths):
        deltatau = np.zeros_like(self.dist_matrix)
        #print(paths)

        for path in paths:
            #        ( origins ,  dests  )
            deltatau[(path[:-1], path[1:])] += 1

        return deltatau


    def update_tau(self, tau_matrix, deltatau, rho):
        return (1-rho)*tau_matrix + deltatau


    def plot_tau(self, ax, tau_matrix):
        ax.clear()

        showtau = tau_matrix / tau_matrix.sum(axis=1).reshape((-1,1))  # normalize each row (from origin `row`, what is probability of traversing to dest `col`)
        im = ax.imshow(showtau, cmap="hot", vmin=0)

        plt.show(block=False)
        plt.pause(1e-12)


    def single_trial(self, tau_matrix, greedy=False):
        #return self.single_trial_stochastic(tau_matrix, greedy)
        #return self.single_trial_sequential(tau_matrix, greedy)
        return self.single_trial_hybrid(tau_matrix, greedy)


    def solve(self):

        print(("K     = {:d}\n"
              +"alpha = {:.3f}\n"
              +"beta  = {:.3f}\n"
              +"gamma = {:.3f}\n"
              +"rho   = {:.3f}").format(self.K, self.alpha, self.beta, self.gamma, self.rho))

        per_trial_paths = [[] for _ in range(self.K)]
        per_trial_scores = [0 for _ in range(self.K)]
        tau_matrix = np.zeros_like(self.dist_matrix)


        nnpaths, nnscores = self.single_trial(tau_matrix, greedy=True)
        total_nnscore = self.total_score(nnscores)
        self.best_score = total_nnscore
        tau_matrix[:,:] = self.K * (total_nnscore / self.max_score)**2  # scale by K since we add K deltataus to tau later
        deltatau = np.zeros_like(tau_matrix)
        print("Greedy score:", total_nnscore)
        self.write_solution_to_file(nnpaths, "solutions/{}/{:09d}.txt".format(self.prob_name[0], int(total_nnscore)))

        fig, ax = plt.subplots()
        plt.ion()
        self.plot_tau(ax, tau_matrix)  # watch tau evolve over time
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        plt.tight_layout()

        pool = mp.Pool(processes=self.cpus)

        while 1:  # goes indefinitely... kill process when you think it's done

            deltatau[:,:] = 0

            results = pool.starmap_async(job_for_pool, [(self, tau_matrix)]*self.K)  # parallelization -- runs jobs on all but one of your cpu cores

            for k,(paths,score) in enumerate(results.get()):
                per_trial_paths[k] = paths
                per_trial_scores[k] = score
                deltatau += self.calc_dtau(paths) * (score / self.max_score)**2  # the square emphasizes the difference between different scores more dramatically

            tau_matrix = self.update_tau(tau_matrix, deltatau, rho)
            self.plot_tau(ax, tau_matrix)

            top_scorer = np.argmax(per_trial_scores)
            top_sched = per_trial_paths[top_scorer]
            top_score = per_trial_scores[top_scorer]
            print("Top score this round:", top_score)
            if top_score >= self.best_score:
                self.best_score = top_score
                self.write_solution_to_file(top_sched, "solutions/{}/{:09d}.txt".format(self.prob_name[0], int(top_score)))


def job_for_pool(m, tau_matrix):
    paths, scores = m.single_trial(tau_matrix)  # step through chunks of time, assigning randomly to travellers who become available in this chunk
    score = m.total_score(scores)

    return paths, score


def gen_pickle(fname):

    with open("inputs/"+fname+".in", "r") as f:
        lines = f.readlines()

    n_rows, n_cols, n_travellers, n_locs, bonus, T = \
                                                list(map(int, lines[0].split()))
    tmp = np.zeros((n_locs+1, 6))
    for i,line in enumerate(lines[1:]):
        tmp[i+1,:] = np.array(list(map(int, line.split())))

    start_row = tmp[:,0]
    start_col = tmp[:,1]
    end_row = tmp[:,2]
    end_col = tmp[:,3]

    dij = np.abs(end_row.reshape((-1,1)) - start_row) \
        + np.abs(end_col.reshape((-1,1)) - start_col)  # one(ish)-liner -- pretty nifty, eh?

    w = np.diag(dij) # dist from end to start of same ride

    si = tmp[:,4]
    fi = tmp[:,5] - w  # convert latest finishes into latest starts

    with open("inputs/"+fname+".pickle", "wb") as f:
        pickle.dump(n_travellers, f) # number in fleet
        pickle.dump(dij, f) # dist. from end of ride i to beg. of ride j
        pickle.dump(w, f) # worth (length) of ride i
        pickle.dump(si, f) # earliest start
        pickle.dump(fi, f) # latest start
        pickle.dump(bonus, f) # bonus value
        pickle.dump(T, f) # last moment in time

    print("generated pickle for "+fname)


def repickle_all(input_names):
    for name in input_names:
        gen_pickle(name)

if __name__=="__main__":

    #repickle_all(input_names)

    in_name = input_names[0] # change this index to change problem being solved
    gr = (np.sqrt(5)+1)/2

    # can't seem to find good param settings ...
    Kpc   = 1         # Kpc   is the number of jobs per core
    alpha = 1/gr      # alpha bigger puts more emphasis on variation in tau -- more focused on successes in previous rounds
    beta  = gr        # beta  bigger puts more emphasis on variation in eta -- more focused on nearest-neighbors connections (in space and time)
    gamma = 1 # gamma in range (0,1] assigns a percentage to the number of travellers who are given an assignment per round. higher gamma means more even assignment across travellers
    rho   = 1/gr**8   # rho   specifies the evaporation rate of pheromones. smaller rho exaggerates more the differences between large and small amounts of pheromones

    m = Myrorna(in_name, Kpc, alpha, beta, gamma, rho)
    m.solve()
