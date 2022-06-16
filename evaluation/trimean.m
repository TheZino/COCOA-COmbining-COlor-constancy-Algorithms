function val = trimean(data)

    % (quartile1 + median * 2 + quartile3) / 4

    % compute 25th percentile (first quartile)
    % Q1 = median(data(find(data<median(data))));
    Q1 = prctile(data, 25)

    % compute 50th percentile (second quartile)
    Q2 = median(data);

    % compute 75th percentile (third quartile)
    % Q3 = median(data(find(data>median(data))));
    Q3 = prctile(data, 75);

    val = (Q1 + Q2*2 + Q3) / 4;

end