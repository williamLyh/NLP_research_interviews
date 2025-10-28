

target_sums = [9, 20, 832112321, 56, 26, 16, 11111111]

paired_target_sums = [(i, val) for i, val in enumerate(target_sums)]
sorted_target_sums = sorted(paired_target_sums, key=lambda x: x[1])
sorting_order = [pair[0] for pair in sorted_target_sums]
target_sums = [pair[1] for pair in sorted_target_sums]

# def find_next_prime(primes):
#     if not primes:
#         return 2
#     cand = primes[-1] + 1
#     if cand % 2 == 0:
#         cand += 1  # make odd
#     while True:
#         is_prime = True
#         for p in primes:
#             if p * p > cand:
#                 break  # no need to check beyond sqrt(cand)
#             if cand % p == 0:
#                 is_prime = False
#                 break  # found a divisor, try next candidate
#         if is_prime:
#             return cand
#         cand += 2  # skip even numbers


# primes = []
# cur_prime = 1

# results = []
# target_idx = 0
# while True:
#     next_prime = find_next_prime(primes)
#     print(next_prime)
#     for i in range(4):
#         temp_sum = i*next_prime + (4-i)*cur_prime
#         # print(temp_sum, target_sums[target_idx] if target_idx<len(target_sums) else None)
#         if (target_idx<len(target_sums)) and (target_sums[target_idx] == temp_sum):
#             results.append([cur_prime]*(4-i) + [next_prime]*i)
#             target_idx += 1
#         if (target_idx<len(target_sums)) and (temp_sum > target_sums[target_idx]):
#             results.append(-1)
#             target_idx += 1

#     cur_prime = next_prime
#     primes.append(next_prime)
#     if target_idx==len(target_sums): break






def sieve(n):
    is_prime = [True]*(n+1)
    is_prime[0] = is_prime[1] = False
    for p in range(2, int(n**0.5)+1):
        if is_prime[p]:
            for multiple in range(p*p, n+1, p):
                is_prime[multiple] = False
    return [i for i, val in enumerate(is_prime) if val]


def four_prime_sum(target_sums):
    max_target = max(target_sums)
    primes = sieve(max_target)  
    primes_set = set(primes)
    results = []

    for n in target_sums:
        if n < 8:
            results.append(-1)
            continue
        if n % 2 == 0:
            a, b = 2, 2
            m = n - 4
        else:
            a, b = 2, 3
            m = n - 5
        found = False
        for p in primes:
            if p > m:
                break
            if (m - p) in primes_set:
                results.append([a, b, p, m-p])
                found = True
                break
        if not found:
            results.append(-1)
    return results

results = four_prime_sum(target_sums)


paired_reuslts = zip(sorting_order, results)
paired_reuslts = sorted(paired_reuslts, key=lambda x: x[0])
_, results = zip(*paired_reuslts)
print(results)
print([sum(val) for val in results if val != -1])
# print(results)
# results = sorted(results, key=sorting_order)
# print(next_prime)


