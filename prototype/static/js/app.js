//read in samples.json
d3.json("static/data/samples.json").then(function(data) {
    //console.log(data);
    samples = data.samples;
    console.log(samples);

    // var names = data.names;
    // console.log(names);
    // var sampleValues = data.map(s =>  s.sample_values);
    // var samples = data.samples;
    // console.log(samples[id]);

    // var names = samples.map(n => n.id);
    // console.log(names);

    // var otuIds = samples.map(o => o.otu_ids);
    // console.log(otuIds);

    // var sampleValues = samples.map(sv => sv.sample_values);
    // console.log(sampleValues);

    // var otuLabels = samples.map(ol => ol.otu_labels);
    // console.log(otuLabels);

  });

// var names = data.names;
// console.log(names);
//   var stock = data.dataset.dataset_code;
//   var startDate = data.dataset.start_date;
