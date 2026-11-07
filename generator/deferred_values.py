"""Head-specific two-stream value reads, without a full projected value bank."""


def form(probability, text_inputs, image_inputs, text_weight, image_weight, mm):
    heads, queries, tokens = probability.shape
    nt, width = text_inputs.shape
    ni = image_inputs.shape[0]
    head_width = text_weight.shape[0] // heads
    if tokens != nt+ni or image_inputs.shape[1] != width or text_weight.shape != image_weight.shape or text_weight.shape != (heads*head_width, width):
        raise ValueError('Deferred head/stream geometry differs')
    text_average = mm(probability[..., :nt], text_inputs, 'weighted-text-input')
    image_average = mm(probability[..., nt:], image_inputs, 'weighted-image-input')
    tw = text_weight.reshape(heads, head_width, width).swapaxes(-1, -2)
    iw = image_weight.reshape(heads, head_width, width).swapaxes(-1, -2)
    text_head = mm(text_average, tw, 'deferred-text-V')
    image_head = mm(image_average, iw, 'deferred-image-V')
    return {'weightedText': text_average, 'weightedImage': image_average,
            'textContribution': text_head, 'imageContribution': image_head, 'H_FP64': text_head+image_head}
