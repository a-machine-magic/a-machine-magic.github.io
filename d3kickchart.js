var n = 17, // number of layers
    m = 137; // number of samples per layer

var margin = {top: 40, right: 100, bottom: 40, left: 75},
    width = 740*2 - margin.left - margin.right,
    height = 500*2 - margin.top - margin.bottom;

var svg = d3.select("#chart-svg").append("svg")
    .attr("width", width + margin.left + margin.right)
    .attr("height", height + margin.top + margin.bottom)
    .append("g")
    .attr("transform", "translate(" + margin.left + "," + margin.top + ")");

d3.csv("sfdata.csv", function (data){
    
    var headers = ["JAN","FEB","MAR","APR","MAY","JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"];
    
    var layers = d3.layout.stack().offset('expand')(headers.map(function(month) {
        return data.map(function(d) {
          return {x: d.YEAR, y: +d[month]};
        });
    }));
console.log(layers);
    var yGroupMax = d3.max(layers, function(layer) { return d3.max(layer, function(d) { return d.y; }); });
    var yStackMax = d3.max(layers, function(layer) { return d3.max(layer, function(d) { return d.y0 + d.y; }); });

    var xScale = d3.scale.ordinal()
        .domain(layers[0].map(function(d) { return d.x; }))
        .rangeRoundBands([25, width], .08);

    var y = d3.scale.linear()
        .domain([0, yStackMax])
        .range([height, 0]);

    var color = d3.scale.ordinal()
        .domain(headers)
        .range(["#6087EA", "#E19BA5", "##5CF18B", "#9AC9CB", "#EAC90C", "#F1492F","#EAD660","#7F776B","#03342D","#cc66ff", "#ff9900"]);
      
    var xAxis = d3.svg.axis()
        .scale(xScale)
        .tickSize(2)
        .tickPadding(6)
        .orient("bottom");

    var yAxis = d3.svg.axis()
        .scale(y)
        .orient("left")
        .tickFormat(d3.format(".2s"));

    var layer = svg.selectAll(".layer")
        .data(layers)
        .enter().append("g")
        .attr("class", "layer")
        .style("fill", function(d, i) { return color(i); });

    var rect = layer.selectAll("rect")
        .data(function(d) { return d; })
        .enter().append("rect")
        .attr("x", function(d) { return xScale(d.x); })
        .attr("y", height)
        .attr("width", xScale.rangeBand())
        .attr("height", 0);

    rect.transition()
        .delay(function(d, i) { return i * 10; })
        .attr("y", function(d) { return y(d.y0 + d.y); })
        .attr("height", function(d) { return y(d.y0) - y(d.y0 + d.y); });

    //********** AXES ************
    svg.append("g")
        .attr("class", "x axis")
        .attr("transform", "translate(0," + height + ")")
        .call(xAxis)
        // .selectAll("text").style("text-anchor", "end")
            .attr("dx", "-.8em")
            .attr("dy", ".15em")
            .attr("transform", function(d) {
                  return "rotate(-45)" 
                });
    
    svg.append("g")
        .attr("class", "y axis")
        .attr("transform", "translate(20,0)")
        .call(yAxis)
      // .append("text")
      //   .attr("transform", "rotate(-90)")
      //   .attr({"x": -150, "y": -70})
      //   .attr("dy", ".75em")
      //   // .style("text-anchor", "end")
      //   .text("Degrees Celcius");

    svg.append("text")
        .attr("x", (width / 2))             
        .attr("y", (margin.top -55))
        .attr("text-anchor", "middle")  
        .style("font-size", "25px")  
        .text(" San Francisco Average Monthly Temperature: 1880-2016");

    var legend = svg.selectAll(".legend")
        .data(headers.slice())
            .enter().append("g")
            .attr("class", "legend")
            .attr("transform", function(d, i) { return "translate(+90," + i * 20 + ")"; });
       
        legend.append("rect")
            .attr("x", width - 18)
            .attr("width", 18)
            .attr("height", 18)
            .style("fill", color);
    
        legend.append("text")
              .attr("x", width - 24)
              .attr("y", 9)
              .attr("dy", ".35em")
              .style("text-anchor", "end")
              .text(function(d) { return d;  });


    d3.selectAll("input").on("change", change);

    var timeout = setTimeout(function() {
      d3.select("input[value=\"grouped\"]").property("checked", true).each(change);
    }, 2000);

    function change() {
      clearTimeout(timeout);
      if (this.value === "grouped") transitionGrouped();
      else transitionStacked();
    }

    function transitionGrouped() {
      y.domain([0, yGroupMax]);

      rect.transition()
          .duration(500)
          .delay(function(d, i) { return i * 10; })
          .attr("x", function(d, i, j) { return xScale(d.x) + xScale.rangeBand() / n * j; })
          .attr("width", xScale.rangeBand() / n)
        .transition()
          .attr("y", function(d) { return y(d.y); })
          .attr("height", function(d) { return height - y(d.y); });

      rect.on("mouseover", function() { tooltip.style("display", null); })
        .on("mouseout", function() { tooltip.style("display", "none"); })
        .on("mousemove", function(d) {
          var xPosition = d3.mouse(this)[0] - 15;
          var yPosition = d3.mouse(this)[1] - 25;
          tooltip.attr("transform", "translate(" + xPosition + "," + yPosition + ")");
          tooltip.select("text").text("hello world");
        });
    };

    function transitionStacked() {
      y.domain([0, yStackMax]);

      rect.transition()
          .duration(500)
          .delay(function(d, i) { return i * 10; })
          .attr("y", function(d) { return y(d.y0 + d.y); })
          .attr("height", function(d) { return y(d.y0) - y(d.y0 + d.y); })
        .transition()
          .attr("x", function(d) { return xScale(d.x); })
          .attr("width", xScale.rangeBand());

      rect.on("mouseover", function() { tooltip.style("display", null); })
        .on("mouseout", function() { tooltip.style("display", "none"); })
        .on("mousemove", function(d) {
          var xPosition = d3.mouse(this)[0] - 15;
          var yPosition = d3.mouse(this)[1] - 25;
          console.log(xPosition);
          tooltip.attr("transform", "translate(" + xPosition + "," + yPosition + ")");
          tooltip.select("text").text("hello world");
        });
    };

    var tooltip = svg.append("g")
        .attr("class", "tooltip");
        
    // tooltip.append("rect")
    //     .attr("width", 30)
    //     .attr("height", 20)
    //     .attr("fill", "red")
    //     .style("opacity", 0.5);
     
    tooltip.append("text")
        .attr("x", 15)
        .attr("dy", "1.2em")
        .style("text-anchor", "middle")
        .attr("font-size", "12px")
        .attr("font-weight", "bold"); });