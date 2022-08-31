export function getFRClass(targetEmbeddings, searchEmbeddings, options) {
  const euclideanDistance = (embeddings1, embeddings2) => {
    let embeddingSum = 0;

    for (let i = 0; i < embeddings1.length; i++) {
      embeddingSum += Math.pow((embeddings1[i] - embeddings2[i]), 2);
    }

    return Math.sqrt(embeddingSum);
  };

  const cosineDistance = (embeddings1, embeddings2) => {
    let dotSum = 0;
    let norm1 = 0;
    let norm2 = 0;

    for (let i = 0; i < embeddings1.length; i++) {
      dotSum = dotSum + embeddings1[i] * embeddings2[i];
      norm1 = norm1 + Math.pow(embeddings1[i], 2);
      norm2 = norm2 + Math.pow(embeddings2[i], 2);
    }

    norm1 = Math.sqrt(norm1);
    norm2 = Math.sqrt(norm2);
    return 1 - dotSum / (norm1 * norm2);
  };

  // Embeddings L2-Normalization
  const l2Normalization = (embeddings) => {
    // norm(L2) = (|x0|^2 + |x1|^2 + |x2|^2 + |xi|^2)^1/2
    let embeddingSum = 0;

    for (let i = 0; i < embeddings.length; i++) {
      if (embeddings[i] !== 0) {
        embeddingSum = embeddingSum + Math.pow(Math.abs(embeddings[i]), 2);
      }
    }

    const L2 = Math.sqrt(embeddingSum);
    const embeddingsNorm = new Float32Array(embeddings.length);

    for (let i = 0; i < embeddings.length; i++) {
      if (embeddings[i] !== 0) {
        embeddingsNorm[i] = (embeddings[i] / L2).toFixed(10);
      } else {
        embeddingsNorm[i] = 0;
      }
    }

    return embeddingsNorm;
  };

  const results = [];
  const distanceMap = new Map();

  for (let i = 0; i < targetEmbeddings.length; i++) {
    for (let j = 0; j < searchEmbeddings.length; j++) {
      // Set default status 'unknown' as 'X'
      results[j] = 'X';
      let distance;
      if (options.distanceMetric === 'euclidean') {
        const [...targetEmbeddingsTmp] =
            Float32Array.from(l2Normalization(targetEmbeddings[i]));
        const [...searchEmbeddingsTmp] =
            Float32Array.from(l2Normalization(searchEmbeddings[j]));
        distance = euclideanDistance(targetEmbeddingsTmp, searchEmbeddingsTmp);
      } else if (options.distanceMetric === 'cosine') {
        distance = cosineDistance(targetEmbeddings[i], searchEmbeddings[j]);
      }
      if (!distanceMap.has(j)) distanceMap.set(j, new Map());
      distanceMap.get(j).set(i, distance);
    }
  }

  console.dir(distanceMap);

  for (const key1 of distanceMap.keys()) {
    let num = null;
    let minDis = null;
    for (const [key2, value2] of distanceMap.get(key1).entries()) {
      if (minDis == null) {
        num = key2;
        minDis = value2;
      } else {
        if (minDis > value2) {
          num = key2;
          minDis = value2;
        }
      }
    }

    if (results[key1] === 'X' && minDis < options.threshold) {
      results[key1] = parseInt(num) + 1;
    }
  }

  return results;
}
