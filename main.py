import numpy as np
import pickle
import matplotlib.pyplot as plt
import multiprocessing as mp


cpus = mp.cpu_count() - 1  # -1 to save one core for normal operations in the computer
print("Available processes:", cpus)


input_names = ["a_example",
               "b_should_be_easy",
               "c_no_hurry",
               "d_metropolis",
               "e_high_bonus"]


def load_pickle(prob_name):
    with open("inputs/"+prob_name+".pickle", "rb") as f:
        n   = pickle.load(f)
        dij = pickle.load(f)
        w   = pickle.load(f)
        si  = pickle.load(f)
        fi  = pickle.load(f)
        b   = pickle.load(f)
        T   = pickle.load(f)
    print("loaded pickle for "+prob_name)
    return n, dij, w, si, fi, b, T


def write_solution_to_file(paths, fname):

    with open(fname, "w") as f:
        for i,path in enumerate(paths):

            outstr = str(len(path)-1)
            for stop in path[1:]:
                outstr += " {:d}".format(int(stop-1))
            outstr += "\n"

            f.write(outstr)

    print("solution written to file "+fname)


def calc_score(tb4s, tb4e, dists_to_rides, lengths_of_rides, bonus):
    earns_bonus = np.greater_equal(tb4s, dists_to_rides) # if it takes less time to get there than the time before it starts
    not_too_late = np.greater_equal(tb4e, dists_to_rides) # " " " " " " " " " " ends

    return not_too_late * (lengths_of_rides + bonus * earns_bonus) # zero where opportunity is past


def time_b4_start(earliest_start, t):
    return earliest_start - t


def time_b4_end(latest_start, t):
    return latest_start - t


def time_busy(earliest_start, dist_to_ride, length_of_ride, t):
    tb4s = time_b4_start(earliest_start, t)

    return max(dist_to_ride, tb4s) + length_of_ride # if we arrive too early, we still have to wait until the start


def calc_eta(not_tabu, earliest_starts, latest_starts, dists_to_rides,
             lengths_of_rides, bonus, t):
    tb4s = time_b4_start(earliest_starts, t)
    tb4e = time_b4_end(latest_starts, t)

    score = calc_score(tb4s, tb4e, dists_to_rides, lengths_of_rides, bonus)
    not_excluded = np.array(
                        [x in not_tabu for x in range(len(lengths_of_rides))])

    #numer1 = np.power(score, 2) * not_excluded  # zero if pos is in tabu list
    numer1 = score * not_excluded  # zero if pos is in tabu list
    if numer1.sum()==0:
        return 0, True

    #numer2 = 1 + np.maximum(0, tb4e - dists_to_rides) # scale by time before end
    denom1 = 1 + dists_to_rides # scale by time taken to get to start
    denom2 = 1 + np.maximum(0, tb4s - dists_to_rides) # scale by time waiting
    #denom2 = 1 + np.abs(tb4s - dists_to_rides) # scale by time wasted
    #denom3 = 1 + (tb4e - tb4s) # scale by size of window of opportunity

    #return numer1/(denom1*denom2*denom3), False
    #return (numer1*numer2)/(denom1*denom2), False
    return numer1/(denom1*denom2), False


def choose_next_step(tau, eta, alpha, beta):
    r = np.random.rand()

    p = np.power(tau, alpha) * np.power(eta, beta)
    intervals = np.cumsum(p)/p.sum()  # essentially roulette-wheel selection
    chc = int(np.where(r - intervals <= 0)[0][0])

    return chc


def single_trial(N, dist_matrix, start_locations, earliest_starts,
                 latest_starts, lengths_of_rides, bonus, T, tau_matrix, alpha,
                 beta, gamma, greedy=False):
    if greedy:
        gamma = 1
    paths = [[0] for _ in range(N)]  # store path taken per traveller
    scores = [[0] for _ in range(N)]  # store score for each step of each traveller
    avail = np.zeros(N, dtype=np.uint32)  # stores time at which each traveller becomes available
    locs = start_locations.copy().astype(np.uint32) # records locations of travellers during their travels
    tabu = set(locs)
    not_tabu = set(range(dist_matrix.shape[0])) - tabu  # store set of nodes that are still available to travellers
    blacklist = set()

    lnt = len(not_tabu)  # keep track of number of available nodes -- if it stagnates, exit loop and return
    clnt = 0
    while clnt < N/gamma and len(blacklist) < N and lnt > 0:
        last_lnt = len(not_tabu)

        travellers = set(np.random.choice(N, size=int(gamma*N), replace=False)) # select some number of travellers randomly
        travellers -= blacklist  # remove blacklisted travellers

        for trv in travellers:  # give each traveller a chance to take a step

            t = avail[trv]
            current_pos = locs[trv]
            dists_to_rides = dist_matrix[current_pos,:]  # get distances to each other node

            eta, cont_flag = calc_eta(not_tabu, earliest_starts, latest_starts,
                                      dists_to_rides, lengths_of_rides, bonus,
                                      t)  # get visibility of each node
            if cont_flag:  # if no nodes are available for whatever reason, skip this traveller going forward (big speedup with this!)
                blacklist.add(trv)
                continue

            if greedy:
                next_pos = int(np.argmax(eta))  # greedy search
            else:
                tau = tau_matrix[current_pos,:]
                next_pos = choose_next_step(tau, eta, alpha, beta)  # select next step probabilistically according to ACO algorithm

            busy = time_busy(earliest_starts[next_pos],
                             dists_to_rides[next_pos],
                             lengths_of_rides[next_pos], t)  # the time that the traveller is occupied before becoming available again
            avail[trv] += int(busy)

            paths[trv].append(next_pos)
            locs[trv] = next_pos

            wait = time_b4_start(earliest_starts[next_pos], t) \
                                    - dists_to_rides[next_pos]
            late = time_b4_end(latest_starts[next_pos], t) \
                                    - dists_to_rides[next_pos]
            score = calc_score(wait, late, dists_to_rides[next_pos],
                               lengths_of_rides[next_pos], bonus)  # calculate score attained by step
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


def single_trial_sequential(N, dist_matrix, start_locations, earliest_starts,
                 latest_starts, lengths_of_rides, bonus, T, tau_matrix, alpha,
                 beta, gamma, greedy=False):
    if greedy:
        gamma = 1
    paths = [[0] for _ in range(N)]  # store path taken per traveller
    scores = [[0] for _ in range(N)]  # store score for each step of each traveller
    avail = np.zeros(N, dtype=np.uint32)  # stores time at which each traveller becomes available
    locs = start_locations.copy().astype(np.uint32) # records locations of travellers during their travels
    tabu = set(locs)
    not_tabu = set(range(dist_matrix.shape[0])) - tabu  # store set of nodes that are still available to travellers

    for t in range(T):

        travellers = np.random.permutation(np.where(avail <= t)[0]) # select all travellers who are available now and randomize the order they can make their choice

        for trv in travellers:  # give each traveller a chance to take a step

            current_pos = locs[trv]
            dists_to_rides = dist_matrix[current_pos,:]  # get distances to each other node

            eta, cont_flag = calc_eta(not_tabu, earliest_starts, latest_starts,
                                      dists_to_rides, lengths_of_rides, bonus,
                                      t)  # get visibility of each node
            if cont_flag:  # if no nodes are available for whatever reason, skip this traveller and set availability to past the deadline
                avail[trv] = T
                continue

            if greedy:
                next_pos = int(np.argmax(eta))  # greedy search
            else:
                tau = tau_matrix[current_pos,:]
                next_pos = choose_next_step(tau, eta, alpha, beta)  # select next step probabilistically according to ACO algorithm

            busy = time_busy(earliest_starts[next_pos],
                             dists_to_rides[next_pos],
                             lengths_of_rides[next_pos], t)  # the time that the traveller is occupied before becoming available again
            avail[trv] += int(busy)

            paths[trv].append(next_pos)
            locs[trv] = next_pos

            wait = time_b4_start(earliest_starts[next_pos], t) \
                                    - dists_to_rides[next_pos]
            late = time_b4_end(latest_starts[next_pos], t) \
                                    - dists_to_rides[next_pos]
            score = calc_score(wait, late, dists_to_rides[next_pos],
                               lengths_of_rides[next_pos], bonus)  # calculate score attained by step
            scores[trv].append(score)

            not_tabu.remove(next_pos)
            tabu.add(next_pos)

    return paths, scores


def total_score(scores):
    return sum([sum(per_trv) for per_trv in scores])


def calc_dtau(paths, shape):
    deltatau = np.zeros(shape)

    for path in paths:
        #        ( origins ,  dests  )
        deltatau[(path[:-1], path[1:])] += 1

    return deltatau


def update_tau(tau_matrix, deltatau, rho):
    return (1-rho)*tau_matrix + deltatau


def plot_tau(ax, tau_matrix):
    ax.clear()

    showtau = tau_matrix / tau_matrix.sum(axis=1).reshape((-1,1))  # normalize each row (from origin `row`, what is probability of traversing to dest `col`)
    im = ax.imshow(showtau, cmap="hot", vmin=0)

    plt.show(block=False)
    plt.pause(1e-12)


def solve(prob_name, K, alpha, beta, gamma, rho):
    N, dist_matrix, lengths_of_rides, earliest_starts, latest_starts, bonus, T =\
                                                          load_pickle(prob_name)
    K   *= cpus  # scale so that the original setting describes amount of work *per worker*
    #rho *= K / dist_matrix.shape[0]  # scale so that it reflects the possible coverage (K) and the range (dij.shape[0]) of the nodes being considered

    print(("K     = {:d}\n"
          +"alpha = {:.3f}\n"
          +"beta  = {:.3f}\n"
          +"gamma = {:.3f}\n"
          +"rho   = {:.3f}").format(K, alpha, beta, gamma, rho))

    start_locations = np.zeros(N, dtype=np.uint32)  # they all start at node 0
    per_trial_paths = [[] for _ in range(K)]
    per_trial_scores = [0 for _ in range(K)]
    tau_matrix = np.zeros_like(dist_matrix)
    max_score = lengths_of_rides.sum() + bonus*len(lengths_of_rides)  # theoretical maximum score -- in general, there will not be possible solutions that can achieve this
    print("Max score:", max_score)

    nnpaths, nnscores = single_trial(N, dist_matrix, start_locations,
                                     earliest_starts, latest_starts,
                                     lengths_of_rides, bonus, T, tau_matrix,
                                     alpha, beta, gamma, greedy=True)
    total_nnscore = total_score(nnscores)
    best_score = total_nnscore
    tau_matrix[:,:] = K * (total_nnscore / max_score)**2  # scale by K since we add K deltataus to tau
    deltatau = np.zeros_like(tau_matrix)
    print("Greedy score:", total_nnscore)
    write_solution_to_file(nnpaths, prob_name[0]+"_solution.txt")

    fig, ax = plt.subplots()
    plt.ion()
    plot_tau(ax, tau_matrix)  # watch tau evolve over time
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    plt.tight_layout()

    args = (N, dist_matrix, start_locations, earliest_starts, latest_starts,
            lengths_of_rides, bonus, T, tau_matrix, alpha, beta, gamma)
    pool = mp.Pool(processes=cpus)

    while 1:  # goes indefinitely... kill process when you think it's done

        deltatau[:,:] = 0

        results = pool.starmap_async(job_for_pool, [args]*K)  # parallelization -- runs jobs on all but one of your cpu cores

        for k,(paths,score) in enumerate(results.get()):
            per_trial_paths[k] = paths
            per_trial_scores[k] = score
            deltatau += calc_dtau(paths, tau_matrix.shape) \
                                                        * (score / max_score)**2  # the square emphasizes the difference between different scores more dramatically

        tau_matrix = update_tau(tau_matrix, deltatau, rho)
        plot_tau(ax, tau_matrix)

        top_scorer = np.argmax(per_trial_scores)
        top_sched = per_trial_paths[top_scorer]
        top_score = per_trial_scores[top_scorer]
        print("Top score this round:", top_score)
        if top_score >= best_score:
            best_score = top_score
            write_solution_to_file(top_sched, prob_name[0]+"_solution.txt")


def job_for_pool(N, dist_matrix, start_locations, earliest_starts,
                 latest_starts, lengths_of_rides, bonus, T, tau_matrix, alpha,
                 beta, gamma):
    paths, scores = single_trial_sequential(N, dist_matrix, start_locations,
    #paths, scores = single_trial(N, dist_matrix, start_locations,
                                 earliest_starts, latest_starts,
                                 lengths_of_rides, bonus, T, tau_matrix, alpha,
                                 beta, gamma, greedy=False)
    score = total_score(scores)

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

    in_name = input_names[1] # change this index to change problem being solved

    # can't seem to find good param settings ...
    Kpc   = 20      # Kpc   is the number of jobs per core
    alpha = 2       # alpha bigger puts more emphasis on variation in tau -- more focused on successes in previous rounds
    beta  = 2       # beta  bigger puts more emphasis on variation in eta -- more focused on nearest-neighbors connections (in space and time)
    gamma = 0.8     # gamma in range (0,1] assigns a percentage to the number of travellers who are given an assignment per round. higher gamma means more even assignment across travellers
    rho   = 0.1     # rho   specifies the evaporation rate of pheromones. smaller rho exaggerates more the differences between large and small amounts of pheromones

    solve(in_name, Kpc, alpha, beta, gamma, rho)
