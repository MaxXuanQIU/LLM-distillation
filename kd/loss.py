"""Exact full-vocabulary forward KL with checkpointed output projections."""
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


def forward_kl(student_logits, teacher_logits, temperature=1.0):
    s = F.log_softmax(student_logits.float() / temperature, dim=-1)
    t = F.log_softmax(teacher_logits.float() / temperature, dim=-1)
    return F.kl_div(s, t, reduction="sum", log_target=True) * temperature ** 2


def response_loss(student, teacher, ids, attention, labels, chunk_size=32, temperature=1.0,
                  *, equal_answers=False):
    s_hidden = student.model(input_ids=ids, attention_mask=attention, use_cache=False).last_hidden_state
    mask = labels[:, 1:] != -100
    sh = s_hidden[:, :-1][mask]
    targets = labels[:, 1:][mask]
    if not targets.numel():
        raise ValueError("A training batch contains no completion tokens")
    if equal_answers:
        if teacher is None or not mask.any(dim=1).all():
            raise ValueError("Answer-weighted KL requires a teacher and nonempty answers")
        weights = mask.sum(dim=1).float().reciprocal()[:, None].expand_as(mask)[mask]
    th = None
    if teacher is not None:
        with torch.no_grad(), torch.autocast(teacher.device.type, dtype=torch.bfloat16,
                                            enabled=teacher.device.type == "cuda"):
            teacher_hidden = teacher.model(input_ids=ids.to(teacher.device),
                attention_mask=attention.to(teacher.device), use_cache=False).last_hidden_state
            th = teacher_hidden[:, :-1][mask.to(teacher.device)]

    def ce_chunk(h, y):
        return F.cross_entropy(student.lm_head(h).float(), y, reduction="sum")

    def kl_chunk(h, t, token_weights=None):
        sl = student.lm_head(h)
        with torch.no_grad():
            tl = teacher.lm_head(t).to(sl.device)
        if token_weights is not None:
            s = F.log_softmax(sl.float() / temperature, dim=-1)
            t = F.log_softmax(tl.float() / temperature, dim=-1)
            terms = F.kl_div(s, t, reduction="none", log_target=True).sum(dim=-1)
            return (terms * token_weights).sum() * temperature ** 2
        return forward_kl(sl, tl, temperature)

    total = sh.new_zeros((), dtype=torch.float32)
    for start in range(0, len(sh), chunk_size):
        stop = start + chunk_size
        if teacher is None:
            part = checkpoint(ce_chunk, sh[start:stop], targets[start:stop], use_reentrant=False)
        elif equal_answers:
            part = checkpoint(kl_chunk, sh[start:stop], th[start:stop], weights[start:stop],
                              use_reentrant=False)
        else:
            part = checkpoint(kl_chunk, sh[start:stop], th[start:stop], use_reentrant=False)
        total = total + part
    return total, len(targets)
