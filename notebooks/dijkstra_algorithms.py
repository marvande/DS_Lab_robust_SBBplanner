import numpy as np
import pandas as pd
import networkx as nx
from scipy import stats
from heapq import heappush, heappop
from itertools import count

MAX_TRIP_DURATION = 2 #duration in hour

def minute_to_string(m):
    hour, minute = m // 60, m - 60*(m//60)
    time_string = '{:02}:{:02}'.format(int(hour), int(minute))
    
    return time_string

def convertToMinute(s):
    h, m = s.split(':')
    h,m = int(h), int(m)
    
    return h*60+m

def normal_dijkstra(G, first_source, paths=None, cutoff=None, last_target=None):
    
    G_succ = G.succ if G.is_directed() else G.adj
    paths = {first_source: [first_source]}

    push = heappush
    pop = heappop
    dist = {}  # dictionary of final distances
    
    # dictionnary of wthether it's the first time a node is visited
    seen = {first_source: 0}

    c = count()
    fringe = []  # use heapq with (distance,label) tuples
    push(fringe, (0, next(c), first_source))
    
    while fringe:
        #take the node to look at: 
        (d, _, source) = pop(fringe)
        
        # check if node has already been looked at: 
        if source in dist:
            continue  # already searched this node.
        
        # update the distance of the node
        dist[source] = d
        
        #stop if the node we look at is the target obviously
        if source == last_target:
            break
            
        # Look at all direct descendents from the source node: 
        for target, edges in G_succ[source].items():
            # Because it's a multigraph, need to look at all edges between two nodes:
            for edge_id in edges:
                
                # Get the duration between two nodes:
                cost = G.get_edge_data(source, target, edge_id)['duration']
                
                if cost is None:
                        continue
                
                # Add the weight to the current distance of a node
                current_dist = dist[source] + cost
                
                # if target has already been visited once and has a final distance:
                if target in dist:
                        # if we find a distance smaller than the actual distance in dic
                        # raise error because dic distances contains only final distances
                        if current_dist < dist[target]:
                            raise ValueError('Contradictory paths found:',
                                             'negative weights?')
                # either node node been seen before or the current distance is smaller than the 
                # proposed distance in seen[target]:
                elif target not in seen or current_dist < seen[target]:
                    # update the seen distance
                    seen[target] = current_dist
                    # push it onto the heap so that we will look at its descendants later
                    push(fringe, (current_dist, next(c), target))
                    
                    # update the paths till target:
                    if paths is not None:
                        paths[target] = paths[source] + [target]
    if paths is not None:
        return (dist, paths)
    return dist

def compute_delay_uncertainty(mean, std, confidence):
    if confidence != None:
        if mean == None or std == None:
            return 0
        
        num_sample = 50 # how many (source, target, train_id, hour) tuple there are
        t_quantile = stats.t(df=num_sample-1).ppf(confidence)
        mean_deviation = t_quantile * std / np.sqrt(num_sample)
        delay = mean_deviation
    else:
        delay = 0
        
    return delay

"""
Old algorithm, useful if given the departure time and not the arrival time
"""
def dijkstra_with_time(G, first_source, arrival_time, last_target=None, confidence=None, 
                       confidence_step=0.01, durations_dicts=None, paths=None):
    G = G.copy()
    departure_time = arrival_time - MAX_TRIP_DURATION*60
    while True:
        # Update durations according to confidence
        if confidence != None:
            if durations_dicts == None:
                raise ValueError('You must pass durations_dicts for the confidence.')
            # Load dict with modifications
            if confidence not in durations_dicts:
                edge_and_data_tuple = zip(G.edges(keys=True), 
                              map(lambda x: x[2], G.edges(data=True)))
                edge_and_data_tuple = filter(lambda x: 'mean' in x[1] and 'std' in x[1], edge_and_data_tuple)
                durations_dicts[confidence] = {e: {'duration': data['mean'] + compute_delay_uncertainty(data['mean'], 
                                                                                                        data['std'], 
                                                                                                        confidence)
                                                   if data['mean'] != None and data['std'] != None
                                                   else data['duration']
                                                  } for e, data in edge_and_data_tuple}
            
            # Update graph
            nx.set_edge_attributes(G, durations_dicts[confidence])
        
        
        G_succ = G.succ if G.is_directed() else G.adj

        paths = {first_source: [first_source]}
        e_paths = {first_source: []}

        push = heappush
        pop = heappop
        dist = {}  # dictionary of final distances

        # dictionnary of whether it's the first time a node is visited
        seen = {first_source: departure_time}


        c = count()
        fringe = []  # use heapq with (distance,label) tuples

        #push(fringe, (0, next(c), first_source))
        push(fringe, (departure_time, next(c), first_source))

        while fringe:
            #take the node to look at: 
            (d, _, source) = pop(fringe)
            #print('Looking at node: '+source)

            # check if node has already been looked at: 
            if source in dist:
                continue  # already searched this node

            # update the distance of the node
            dist[source] = d

            #stop if the node we look at is the target obviously
            if source == last_target:
                break

                
            # Look at all direct descendents from the source node: 
            for target, edges in G_succ[source].items():
                # Because it's a multigraph, need to look at all edges between two nodes:
                for edge_id in edges:
                    # Check if walking edge
                    dep_time_edge = G.get_edge_data(source, target, edge_id)['time']
                    if dep_time_edge == -1:
                        walking_edge = True
                        current_trip_id = None
                        dep_time_edge = d
                    else:
                        walking_edge = False
                        current_trip_id = G.get_edge_data(source, target, edge_id)['trip_id']
                        
                    
                    if dep_time_edge >= dist[source]:
                        # Check if edge is feasible (also accoring to confidence)
                        if len(e_paths[source]) >= 1 and not e_paths[source][-1][2]['walk']:
                            last_edge_source, last_edge_target, last_edge_info = e_paths[source][-1]
                            last_delay = compute_delay_uncertainty(last_edge_info['mean'], 
                                                                   last_edge_info['std'], 
                                                                   confidence)
                            # If we make a transport-walk change, add delay to walk time
                            if walking_edge:
                                dep_time_edge += last_delay
                            else:
                                # If we make a transport-transport change, check if we have time
                                if current_trip_id != last_edge_info['trip_id']\
                                and dep_time_edge < dist[source] + 2 + last_delay:
                                    continue

                        # Get the duration between two nodes:
                        duration_cost = G.get_edge_data(source, target, edge_id)['duration']
                        if duration_cost is None:
                                continue

                        # Add the weight to the current distance of a node
                        current_dist = dep_time_edge + duration_cost

                        # if target has already been visited once and has a final distance:
                        if target in dist:
                                # if we find a distance smaller than the actual distance in dic
                                # raise error because dic distances contains only final distances
                                if current_dist < dist[target]:
                                    raise ValueError('Contradictory paths found:',
                                                     'negative weights?')

                        # either node has been seen before or the current distance is smaller than the 
                        # proposed distance in seen[target]:
                        elif target not in seen or current_dist < seen[target]:
                            # update the seen distance
                            seen[target] = current_dist
                            # push it onto the heap so that we will look at its descendants later
                            push(fringe, (current_dist, next(c), target))

                            # update the paths till target:
                            if paths is not None:
                                edge_dict = G.get_edge_data(source, target, edge_id)
                                
                                edge_dict['walk'] = walking_edge
                                edge_dict['departure_time'] = dep_time_edge
                                
                                e_paths[target] = e_paths[source] + [(source, target, edge_dict)]


        # No path exists
        if  last_target not in e_paths:
            print('Error: No paths to the source')
            return pd.DataFrame(columns=['from', 'from_id', 'to', 'to_id', 'duration', 'total_duration',
                                         'departure_time', 'walk', 'no_change', 'mean_std_null'])

        
        # Validation
        if confidence == None or validate_path(e_paths[last_target], confidence, G):
            break
        else:
            confidence += confidence_step
            
    # Path validated
    if paths is not None:
        nodes_data = G.nodes(data=True)
        arrival_string = minute_to_string(dist[last_target])
        best_path = e_paths[last_target]
        departure_string = minute_to_string(best_path[0][2]['departure_time'])
        print('Going from {} ({}) to {} ({}) in {:.2f} minutes, departure at {}'.format(nodes_data[first_source]['name'],
                                                                                      first_source,
                                                                                      nodes_data[last_target]['name'],
                                                                                      last_target, 
                                                                                      dist[last_target] - departure_time,
                                                                                      minute_to_string(departure_time)))
        
        # Construct best path's data structure
        best_path_df = pd.DataFrame(columns=['from', 'from_id', 'to', 'to_id', 'duration', 'total_duration',
                                          'departure_time', 'walk', 'no_change', 'mean_std_null'])
        last_edge_info = False
        for source, target, edge_info in best_path:
            no_change = ('trip_id' in edge_info                                   # We're in a transport
                         and last_edge_info and 'trip_id' in last_edge_info       # and last edge also
                         and last_edge_info['trip_id'] == edge_info['trip_id'])   # and same trip_id
            mean_std_null = 'trip_id' in edge_info and 'mean' not in edge_info or 'std' not in edge_info
            
            current_path_dict = {'from': nodes_data[source]['name'],
                                 'from_id': source, 
                                 'to': nodes_data[target]['name'], 
                                 'to_id': target, 
                                 'duration': edge_info['duration'], 
                                 'total_duration': dist[target] - departure_time,
                                 'departure_time': minute_to_string(edge_info['departure_time']), 
                                 'walk':edge_info['walk'], 
                                 'no_change': no_change, 
                                 'mean_std_null': mean_std_null}
            best_path_df = best_path_df.append(current_path_dict, ignore_index=True)
            last_edge_info = edge_info
        
        with pd.option_context('display.max_rows', None, 
                               'display.max_columns', None, 
                               'display.max_colwidth', 15,
                               'display.expand_frame_repr', False):
            print(best_path_df)
        return best_path_df
    raise ValueError('Should not be here')
    return dist

def dijkstra_reversed(G, first_source, arrival_time, last_target, confidence=None, 
                       confidence_step=0.01):
    G = G.copy()
    departure_time = -arrival_time
    
    confidence_dict = {'0.90':'p_90','0.91':'p_91','0.92':'p_92','0.93':'p_93','0.94':'p_94','0.95':'p_95',
                       '0.96':'p_96','0.97':'p_97','0.98':'p_98','0.99':'p_99'}

    # inverse direction 
    temp = first_source
    first_source = last_target
    last_target = temp
        
    if not G.is_directed():
        raise ValueError('Input graph is not directed while it should be.')

    G_succ = G.succ 
        
    # paths stores the nodes in dijkstra's shortest path
    paths = {first_source: [first_source]}
    # stores the edges in dijkstra's shortest path
    e_paths = {first_source: []}
        
    # dictionary of final distances to nodes
    dist = {}  
        
    # dictionnary of whether it's the first time a node is visited
    seen = {first_source: departure_time}

    # use heapq with (distance,label) tuples
    push = heappush
    pop = heappop
    c = count()
    fringe = []  
        
    # push the source as the first node on the heap
    push(fringe, (departure_time, next(c), first_source))
    # while heap not empty
    while True:
        while fringe:
            # take the node to look at: 
            (d, _, source) = pop(fringe)

             # check if node has already been looked at and has a final shortest distance: 
            if source in dist:
                  continue  # already searched this node so go to another

            # take the distance to the node from the heap 
            # source starts with distance = departure_time
            dist[source] = d

            #stop if the source is the last_target. 
            if source == last_target:
                break

            # Look at all direct descendents from the source node: 
            for target, edges in G_succ[source].items():
                # Because it's a multigraph, need to look at all edges between two nodes:
                for edge_id in edges:  
                    
                    dep_time_edge = - G.get_edge_data(source, target, edge_id)['time']

                    # Get the duration between two nodes:
                    duration_cost = G.get_edge_data(source, target, edge_id)['duration']
                     
                    # Check if walking edge: 
                    # walking edges have a departure time of -1
                    if dep_time_edge == 1:
                        walking_edge = True
                        current_trip_id = None
                        
                        # set the departure time to the 
                        # distance to that node as we can leave immediatly
                        dep_time_edge = d + duration_cost                  

                    else:
                        walking_edge = False
                        current_trip_id = G.get_edge_data(source, target, edge_id)['trip_id']


                    if duration_cost is None:
                            raise ValueError('Edge without a duration.')

                    # Add the weight to the current distance to a node

                    current_dist = dep_time_edge           
                    # take only edges that have a departure time bigger 
                    # than the time it takes to get to the node

                    if dep_time_edge - duration_cost < dist[source]:
                            # move on to next edge if it's earlier 
                        continue

                    # Check if edge is feasible (also accoring to confidence)
                    # Check if last edge taken was not a walking edge
                    # Check if there is at least a path of length 1 to the source node 
                    # (e.g. that this node is not the original source)
                    if len(e_paths[source]) >= 1 and not walking_edge:
                        last_edge_source, last_edge_target, last_edge_info = e_paths[source][-1]
                        mean = G.get_edge_data(source, target, edge_id)['mean']
                        std = G.get_edge_data(source, target, edge_id)['std']
                        
                        # now compute delay for current edge because in reverse
                        if str(confidence) in confidence_dict:
                            p = confidence_dict[str(confidence)]
                            last_delay = G.get_edge_data(source, target, edge_id)[p]
                            if last_delay == None: 
                                last_delay = 0
                        else:
                            last_delay = compute_delay_uncertainty(mean, std, confidence)

                        # If we make a transport-> walk change
                        if last_edge_info['walk']:
                            # add delay to departure time of walk as we will leave later
                            dep_time_edge -= last_delay
                        else:
                            # If we make a transport->transport change, check if we have time to change
                            # To change we need that the next connection leaves >= 2 min + delay of transport
                            # If not we cannot take that edge
                            if current_trip_id != last_edge_info['trip_id'] and dep_time_edge - last_delay - 2 - duration_cost < dist[source]:
                                continue                

                    # if target has already been visited once and has a final distance:
                    if target in dist:
                            # if we find a distance smaller than the actual distance in dic
                            # raise error because dic distances contains only final distances
                            if current_dist < dist[target]:
                                raise ValueError('Contradictory paths found:','negative weights?')

                    # either node has been seen before or the current distance is smaller than the 
                    # proposed distance in seen[target]:
                    if target not in seen or current_dist < seen[target]:

                        # update the seen distance
                        seen[target] = current_dist
                        # push it onto the heap so that we will look at its descendants later
                        push(fringe, (current_dist, next(c), target))

                        # update the paths till target:
                        if paths is not None:

                            edge_dict = G.get_edge_data(source, target, edge_id)

                            edge_dict['walk'] = walking_edge
                            edge_dict['departure_time'] = dep_time_edge
                            e_paths[target] = e_paths[source] + [(source, target, edge_dict)]    
        
        # If there is no path to the last_target:
        if  last_target not in e_paths:
            print('Error: No paths to the source')
            return pd.DataFrame(columns=['from', 'from_id', 'to', 'to_id', 'duration', 
                                             'departure_time', 'walk', 'no_change', 'mean_std_null','mean','std'])

            
        # Construct path's data structure
        if paths is not None:
            nodes_data = G.nodes(data=True)
            arrival_string = minute_to_string(dist[last_target])

            #reverse path:
            best_path = []
            for edge in e_paths[last_target][::-1]: 
                edge = (edge[1], edge[0], edge[2])
                best_path.append(edge)           
            num_edges = len(best_path)
            departure_string = minute_to_string(-best_path[0][2]['departure_time'])

            best_path_df = pd.DataFrame(columns=['from', 'from_id', 'to', 'to_id', 'duration',
                                              'departure_time', 'walk', 'no_change', 'mean_std_null', 'mean','std'])
            last_edge_info = False
            for source, target, edge_info in best_path:
                no_change = ('trip_id' in edge_info                                   # We're in a transport
                             and last_edge_info and 'trip_id' in last_edge_info       # and last edge also
                             and last_edge_info['trip_id'] == edge_info['trip_id'])   # and same trip_id
                mean_std_null = 'trip_id' in edge_info and 'mean' not in edge_info or 'std' not in edge_info

                if not mean_std_null:
                    mean = edge_info['mean']
                    std = edge_info['std']
                    if  edge_info['mean'] == None or  edge_info['std'] == None: 
                        mean = edge_info['duration']
                        std = 0
                if 'mean' not in edge_info or 'std' not in edge_info:
                    mean = edge_info['duration']
                    std = 0

                current_path_dict = {'from': nodes_data[source]['name'],
                                     'from_id': source, 
                                     'to': nodes_data[target]['name'], 
                                     'to_id': target, 
                                     'duration': edge_info['duration'], 
                                     'departure_time': minute_to_string(-edge_info['departure_time']), 
                                     'walk':edge_info['walk'], 
                                     'no_change': no_change, 
                                     'mean_std_null': mean_std_null,
                                    'mean':mean,
                                    'std':float(std),
                                    'transport':edge_info['transport'] if not edge_info['walk'] else 'Walk'}
                best_path_df = best_path_df.append(current_path_dict, ignore_index=True)
                last_edge_info = edge_info

            # Validation: 
            if confidence != None and not validate_path_(best_path_df, confidence):
                # else increase confidence by a confidence step and start again: 
                confidence += 0.01
                print(confidence)
                continue
                
            # If path validated, print it
            print('Going from {} ({}) to {} ({}) in {:.2f} minutes, departure at {}, arrival at {}'.format(nodes_data[last_target]['name'],
                                                                                          last_target,
                                                                                          nodes_data[first_source]['name'],
                                                                                          first_source, 
                                                                                          arrival_time + best_path[0][2]['departure_time'],
                                                                                          minute_to_string(-best_path[0][2]['departure_time']),
                                                                                            minute_to_string(-best_path[num_edges-1][2]['departure_time']+
                                                                                                            best_path[num_edges-1][2]['duration'])))
            with pd.option_context('display.max_rows', None, 
                                   'display.max_columns', None, 
                                   'display.max_colwidth', 15,
                                   'display.expand_frame_repr', False):
                    print(best_path_df)
            return best_path_df
    raise ValueError('Should not be here')
    return e_paths[last_target], dist[last_target]

def is_path_valid(path):
    """is_path_valid: returns true if there is time to take all edges, and if 
    when chaning from a connection to another you have at least 2 minutes. """

    last_target = path['from_id'][len(path['from_id'])-1]
    time = convertToMinute(path['departure_time'][0]) + path['duration'][0]
    
    for i in range(1, len(path['from_id'])):
        #in case an edge taken actually left before we got there (only for transport edges, not for walks)
        if not path['walk'][i] and convertToMinute(path['departure_time'][i]) < time:
            print('You miss this connection. Time is {} while this edge leaves at {} from {} to {}'\
                  .format(minute_to_string(time), path['departure_time'][i], path['from'][i], path['to'][i]))
            return False
        
        #in case of change type transport -> trasnport need 2 minutes transfer:
        if not path['no_change'][i] and not path['walk'][i]:
            if not path['walk'][i-1]:
                if convertToMinute(path['departure_time'][i]) < time + 2:
                    print('You do not have time to change to this connection between {} to {} leaving at {}. You arrive at {} and need at least 2 min transfer'\
                          .format(path['from'][i],path['to'][i], path['departure_time'][i], minute_to_string(time)))
                    return False
        
        else: 
            time = convertToMinute(path['departure_time'][i]) + path['duration'][i]
    return True

def validate_path_(path, confidence):
    num_tries = 100
    num_valids = 0
    
    for i in range(num_tries):
        path_copy = path.copy()
        for i in range(len(path['from_id'])):
            #only for transfers etiher to other trains or to walking: 
            if i > 1 and not path['no_change'][i]:
                mean = path['mean'][i-1]
                std = path['std'][i-1]
                # calcluate delay for connection of before:
                if std != 0:
                    delay = np.random.normal(mean, std)
                    if delay <0:
                        delay = 0
                else: delay = 0
                
                # if its between two transports we just add it to trip duration:
                if not path['walk'][i] and not path['walk'][i-1]:
                    path_copy['duration'][i-1] += delay
                
                # transfer from trans to walk:
                if not path['walk'][i-1] and path['walk'][i]:
                    # if a train to a walk is delayed, the walk needs to leave later:
                    #need to leave at the time it takes for the delayed connection to arrive,                     
                    #add duration to transp: 
                    path_copy['duration'][i-1] += delay
                    
                    #delay the start of walk:
                    arrival_of_edge_before = path_copy['duration'][i-1]+convertToMinute(path_copy['departure_time'][i-1])
                    
                    # need to start later:
                    new_dep_time = minute_to_string(arrival_of_edge_before)
                    path_copy['departure_time'][i] = new_dep_time
        
        if is_path_valid(path_copy):
            num_valids += 1
        perc = num_valids/float(num_tries)
    return perc >= confidence