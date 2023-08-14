def intersectInterval(I_1, I_2):
    I = [max(I_1[0], I_2[0]), min(I_1[1], I_2[1])]
    if I[0] > I[1]:  # If the intervals do not intersect
        I = []      # Return an empty interval
    return I
